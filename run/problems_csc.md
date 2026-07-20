# 双臂 MPC 夹取避障：问题记录与技术总结

> 目标：双臂 Franka 机器人在 MPC 闭环控制下，抓取长条形物体（棍子）后，三者作为刚体共同进行避障运动。

---

## 1. 架构概述

```
run/
  main.py           入口，Viser 可视化 + 主循环
  controller.py     DualArmMPCController — MPC 初始化、协同约束、控制循环
  gripper.py        抓取管理器（视觉跟随 + 碰撞附着）
  scene.py          场景加载、Viser 渲染
  utils.py          四元数运算、位姿转换
```

- **MPC 引擎**: `ModelPredictiveControl` (cuRobo 内置)
- **机器人**: `dual_franka_with_camera.yml`（双臂 Franka，14 DoF，手指锁定）
- **协同机制**: `RelativePoseCost`（软约束 + 硬约束，约束双臂相对位姿）
- **碰撞检测**: `SceneCollision` (GPU 加速)

---

## 2. 尝试过的方案

### 2.1 方案 A：AttachmentManager 动态附着

**思路**: 运行时用 `AttachmentManager` 将障碍物转换为碰撞球，写入机器人的 `link_spheres`，使其参与碰撞检测。

**实现**:
- 修改 robot config：将 `attached_object_{left,right}` 的 parent 改为各自的 TCP（`panda_{left,right}_hand_tcp`）
- 增加球槽：`extra_collision_spheres: 32`
- `grasp_obstacle()` 调用 `fit_spheres()` + `update()` 写入双臂 link_spheres
- 通过 `right_offset = left_TCP * right_TCP⁻¹` 补偿 `update()` 硬编码 `tool_frames[0]` 的问题

**核心几何公式**:
```
左臂: world_objects_pose_offset = identity
      obj_to_link = left_TCP_world⁻¹ * I = left_TCP_world⁻¹
      FK: left_TCP_world * sphere_in_left_TCP = sphere_world ✓

右臂: world_objects_pose_offset = left_TCP * right_TCP⁻¹
      obj_to_link = left_TCP⁻¹ * (left_TCP * right_TCP⁻¹) = right_TCP⁻¹
      FK: right_TCP_world * sphere_in_right_TCP = sphere_world ✓
```

**遇到的问题**:

| 问题 | 根因 | 解决方案 | 效果 |
|------|------|----------|------|
| 穿模（附着球穿透世界障碍物） | CUDA graph 在 setup 时捕获，空球状态被 baked in。in-place 写入 link_spheres 后 graph 重放可能读到缓存值。 | `reset_cuda_graph()` + `cold_start_solve()` 重建 graph | 部分解决，仍有残余 |
| 右臂抖动 | `self_collision_ignore` 未覆盖全量 link。右手看到左臂上的附着球，自碰撞代价导致抖动。 | 扩展 `self_collision_ignore` 到所有碰撞 link | 解决 |
| grasp 瞬间抖动 | warm-start 种子基于旧代价图优化，与新代价图不兼容。 | cold_start 从零开始 | 基本解决，有 ~1s 延迟 |
| 无法靠近目标 | `activation_distance=0.05` 过大，手被障碍物推开。 | 调小到 0.02 | 改善 |
| ungrasp 后障碍物不恢复 | graph 重建前未 re-enable，或 graph 缓存了 disable 状态。 | 确保 enable 在 graph rebuild 之前 | 解决 |
| 物体始终附着在左臂 | `AttachmentManager.update()` 硬编码 `ee_link = tool_frames[0]`（左 TCP），无法自由选择附着点。 | 通过 right_offset 公式补偿 | 可行但复杂 |

**结论**: `AttachmentManager` 是为单臂设计的底层 API，用于双臂需大量 workaround。核心矛盾在于 **MPC 的 CUDA graph 与动态 link_spheres 更新之间的不兼容**。

### 2.2 方案 B：Bake 物体到 URDF

**思路**: 仿照 `dual_arm_gpu_mpc` 仓库的做法，将棍子作为 `extra_link` baked 到 robot model 中，规避动态附着的 graph 问题。

**实现**: 创建 `run/demo/robot_with_stick.yml`，添加 `carried_stick` 作为 extra_link，parent 为 `panda_left_hand_tcp`，定义 18 个碰撞球。

**遇到的问题**:

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| 穿模 | `extra_collision_spheres` 在 cuRobo 代码中**覆盖**了 `collision_spheres` 的定义（line 183: `self.collision_spheres[k] = new_spheres`），导致所有球变为 radius=-100 的空球 | 移除 `extra_collision_spheres` 中的 carried_stick，只在 `collision_spheres` 中定义 |
| 球方向错误 | TCP 的 local X 轴对齐双手连线（world Y），而碰撞球定义在 Y 轴 | 改为沿 X 轴排列 |
| 手无法靠近棍端 | 默认姿态双手间距仅 0.41m，而棍长 0.85m，手在棍子 1/3 处 | 需调整默认关节角 |
| EE 姿态不理想 | 默认 TCP 朝向有一定倾斜 | 需 IK 求解新姿态 |

**结论**: Baked-in 方法避免动态 graph 问题，但缺乏灵活性（无法动态抓取/释放），且球配置需精确对齐 kinematic 链。

### 2.3 方案 C：世界障碍物跟随

**思路**: 不用 AttachmentManager。抓取后每步用 `scene_collision.update_obstacle_pose()` 更新物体位置到双手中点，保持物体在世界场景中。

**核心矛盾**:
```
世界障碍物 → 机器人必须避开它
搬运的物体 = 世界障碍物 → 机器人会避开自己正在搬运的东西
→ 左手紧挨物体 → 碰撞检测触发 → 优化器推走
→ 机器人"甩掉"自己搬运的物体
```

**结论**: cuRobo 不支持"对特定 robot link 豁免某个世界障碍物"的机制。此方案不可行。

### 2.4 方案 D：纯视觉跟随（当前 baseline）

**思路**: 完全不碰碰撞附着。抓取时只做视觉反馈（Viser 框变色跟随），物体碰撞体留在原位不动。

**优点**: 零复杂度，无任何 graph/碰撞问题。
**缺点**: 不是真正的"抓取+避障"，被搬运的物体碰撞体还留在原地，可能阻挡机器人。

---

## 3. CUDA Graph 时序问题

```
时间 ────────────────────────────────────►

[1] MPC setup() → cold_start_solve()
    └─ 捕获 CUDA graph（link_spheres 全为空球，radius=-100）

[2] 正常追踪 (step × N)
    └─ warm_start → 重放 graph → FK 读 link_spheres → 碰撞检查

[3] 用户 Grasp
    ├─ AttachmentManager.update() → 写入新球（in-place）
    ├─ ? graph 重放能否看到新值？
    │
    ├─ 理论上：CUDA graph 捕获 kernel launch，重放时读同一内存地址，应可见
    ├─ 实际上：cuRobo rollout 内部可能有中间缓存（FK 输出被 graph 优化冻结）
    └─ 必须 reset_cuda_graph() + cold_start 重新捕获

[4] 后续追踪
    └─ 新 graph 含正确的球 → 避障生效
```

**关键发现**: `reset_cuda_graph()` 需要:
1. CUDA >= 12.0 ✓
2. `curobo_runtime.cuda_graph_reset = True`（在 `__init__.py` 所有 import 之前设置）
3. 之后 `cold_start_solve()` 自动捕获新 graph

---

## 4. 约束机制

### 4.1 cuRobo 内置 Cost 类型

| Cost | 用途 |
|------|------|
| `ToolPoseCost` | EE 位姿追踪 |
| `SceneCollisionCost` | 世界碰撞 |
| `SelfCollisionCost` | 自碰撞 |
| `CSpaceCost` | 关节限位/速度/加速度 |
| `CSpaceDistCost` | 距起始/目标关节距离 |
| `RelativePoseCost` | 双臂相对位姿约束 |
| `CostSupportPolygon` | 平衡/支撑域 |

### 4.2 约束注入方式

```python
# ① 创建 cost config
rel_pose_cfg = RelativePoseCostCfg(
    weight=torch.tensor([50000.0, 5000.0]),
    primary_tool_frame="panda_left_hand_tcp",
    secondary_tool_frame="panda_right_hand_tcp",
    convert_to_binary=True,  # False=软代价, True=硬约束
)

# ② 注入到 rollout config
for rc in optimizer_rollout_configs:
    rc.cost_cfg.relative_pose_cfg = rel_pose_cfg       # 优化代价
    rc.constraint_cfg.relative_pose_cfg = rel_pose_hard # 可行性约束

# ③ 运行时开关
cost = mgr.get_cost("relative_pose")
cost.enable_cost()   # 启用
cost.disable_cost()  # 禁用
```

---

## 5. MPC 热启动问题

```
常规追踪:
  上一帧轨迹 → 偏移一格 → 热启动种子 → 少量迭代收敛 → 平滑

attach 瞬间:
  上一帧轨迹（基于空球优化） → 热启动种子（不合法）
  → 碰撞代价暴增 → 优化器剧烈修正 → 抖动
  
解决:
  cold_start → 从零优化 → 产生合法轨迹 → 后续 warm_start 正常
```

---

## 6. 关键参数

| 参数 | 默认值 | 影响 |
|------|--------|------|
| `activation_distance` | 0.01 | 碰撞预激活距离。太大=无法靠近目标，太小=穿模 |
| `surface_radius` | 0.002 | fit_spheres 球半径。太大=球太胖影响操作，太小=覆盖不足 |
| `num_spheres` | — | 球数量。太少=间隙大穿模，太多=计算量大 |
| `self_collision_ignore` | 部分 | 必须覆盖双臂所有 link，否则一手避让另一手的附着球 |
| `use_cuda_graph` | True | True=快但不感知动态更新，False=慢但正确 |
| `cold_start_iters` | 300 | 从零优化的迭代数。越多越收敛 |
| `warm_start_iters` | 200 | 热启动迭代数。越多越平滑但越慢 |

---

## 7. 外部参考

- **cuRobo 官方高层 API**: `MotionGen.attach_objects_to_robot()` — 需要 Isaac Sim 集成，本地纯库版本无此 API
- **BEHAVIOR-1K**: 完整的 pick-carry-place 示例，使用 `MotionGen.compute_trajectories(attached_obj=...)`
- **dual_arm_gpu_mpc**: 双臂 GPU-MPC，使用双四元数 + nullspace 投影，物体 bake 在 URDF 中
- **cuRobo Discussion #361**: 工具球 + attach 工作流
- **cuRobo Discussion #91**: 批处理规划不支持 per-batch attach/detach
- **IROS 2025 Hierarchical Framework**: 分层规划 + 经典控制，双臂搬运托盘+咖啡

---

## 8. 结论

**cuRobo 纯库版本（无 Isaac Sim）的动态 attach 机制不适配双臂 MPC 闭环控制。** 核心瓶颈是 `AttachmentManager.update()` 的单臂设计 + CUDA graph 的动态更新不兼容。绕过的代价（reparent link、graph rebuild、全量 self_collision_ignore、参数调优）使得方案脆弱且参数敏感。

**可行的替代方向**:
1. 升级到带 Isaac Sim 的版本，使用 `MotionGen` 高层 API
2. 分离规划层和执行层：规划用 MotionPlanner + attach，执行用纯追踪 MPC
3. 将物体 bake 到 URDF，配合协同模式做固定刚体搬运
