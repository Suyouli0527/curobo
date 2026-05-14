# Dual-Arm LEGO Stacking Simulation

基于 **cuRobo v2** + **Viser** 的双臂乐高堆叠仿真。

参考单臂实现：`simple_stacking_v2.py`（位于仓库根目录，约 440 行）。  
双臂机器人配置参考：`curobo/content/configs/robot/dual_ur10e.yml`（两个末端 `tool0` / `tool1`）。

所有脚本放在 `lego_stacking/` 目录下。

---

## 1. 场景配置与几何参数

场景定义从 `lego_stacking/config/` 下的配置文件读取。

### 1.1 坐标系与布局

- **世界坐标系原点**：位于双臂基座中间。
- **双臂布局**：沿 x 轴相向放置，左右臂间距 `0.8 m`。
  - 左臂（arm_left）基座位置：`(-0.4, 0.0, 0.0)`
  - 右臂（arm_right）基座位置：`(0.4, 0.0, 0.0)`
- **Baseplate**：放置在双臂中间，上表面高度为 `0.2 m`。

### 1.2 可配置几何参数（`config/scene_config.json` 或脚本内常量）

```json
{
  "stud_spacing": 0.016,
  "brick_height": 0.0192,
  "baseplate_height": 0.0032
}
```

### 1.3 Baseplate

- 尺寸：`[16 * stud_spacing, 16 * stud_spacing, baseplate_height]`
- 坐标原点：**上表面中心**
- 因此 `baseplate` 的世界坐标 pose 为：
  - position: `[0.0, 0.0, 0.2 - baseplate_height / 2]`
  - quaternion: `[1.0, 0.0, 0.0, 0.0]`

### 1.4 乐高块（Bricks）

- 尺寸：`[N * stud_spacing, M * stud_spacing, brick_height]`
- 坐标原点：**下表面中心**
- 默认 `M = N = 2`（即 2x2 砖块）
- **碰撞**：所有 lego brick **不参与碰撞检测**，机器人与 brick 之间忽略碰撞。

---

## 2. 乐高布局：`layout.json`

`lego_stacking/config/layout.json` 定义所有乐高块的位置和构建顺序。

### 2.1 字段说明

```json
{
    "1": {
        "x": 3,
        "y": 8,
        "z": 1
    },
    "2": {
        "x": 5,
        "y": 8,
        "z": 1
    }
}
```

| 字段 | 说明 |
|------|------|
| `x`, `y`, `z` | 整数网格坐标，以 `stud_spacing`（x/y）和 `brick_height`（z）为单位。对应 baseplate 上的 16x16 网格和层号。 |

### 2.2 坐标转换

砖块在世界坐标系中的位置：

```python
# baseplate 上表面中心的世界坐标
baseplate_surface_x = 0.0
baseplate_surface_y = 0.0
baseplate_surface_z = 0.2

# 砖块中心（下表面中心）
brick_x = baseplate_surface_x + (x - 7.5) * stud_spacing
brick_y = baseplate_surface_y + (y - 7.5) * stud_spacing
brick_z = baseplate_surface_z + (z - 1) * brick_height + brick_height / 2
```

> 注：baseplate 为 16x16 stud，中心列/行索引为 7.5（0-based）。砖块尺寸按默认 2x2 计算时，中心位于 `(x + 0.5) * stud_spacing` 处。建议统一以砖块几何中心为放置目标点。

---

## 3. 脚本架构要求

### 3.1 文件结构

```
lego_stacking/
|-- config/
|   |-- layout.json          # 乐高布局
|   |-- scene_config.json    # 几何参数（stud_spacing, brick_height 等）
|-- scene_builder.py         # 从 config 生成 Scene (Cuboid) 和 Viser objects
|-- dual_arm_planner.py      # 封装双臂 MotionPlanner，管理左右臂 motion planning
|-- stacking_controller.py   # 主状态机：解析 layout，按顺序规划 pick-and-place
|-- main.py                  # 入口脚本
```

### 3.2 核心行为

1. **加载配置**：从 `config/scene_config.json` 读取参数，从 `config/layout.json` 读取砖块列表。
2. **构建场景**：将 baseplate 作为 `Cuboid` 加入 `Scene`。Lego bricks **不加入 collision scene**，因为它们不参与碰撞检测。
3. **解析构建顺序**：按 `z`（层号）排序，从底层到顶层逐层放置。同层内按 ID 顺序。
4. **双臂分配**：左右臂交替执行 pick-and-place，或按空间位置自动分配（左半区用左臂，右半区用右臂）。
5. **碰撞管理**：Lego bricks 不参与碰撞检测，因此无需在 pick/place 前后对 brick 进行禁用/启用操作。仅需在 place 后更新 Viser 中 brick 的视觉位置。
6. **Viser 可视化**：每个砖块以 box 渲染；被抓取时隐藏原砖块，使其跟随末端执行器移动。

### 3.3 Viser 交互界面

参考 `curobo/examples/getting_started/motion_planning.py --visualize` 的交互模式，保留以下元素：

- **拖动目标帧**：在 Viser 3D 视图中拖动末端执行器的控制帧（control frame）来调整 goal pose。
- **拖动障碍物**：baseplate 等场景障碍物支持拖动 reposition。
- **"Next Step" 按钮**：点击后执行下一步 pick 或 place 操作（参考 `simple_stacking_v2.py` 的按钮逻辑）。

**忽略以下元素**：
- **Joint Trajectory 图像**：不显示关节位置/速度/加速度的实时 plot（即不添加 `server.gui.add_image` 形式的 trajectory panel）。

### 3.4 Motion Planning API

- 使用 `MotionPlanner` / `MotionPlannerCfg.create(robot="dual_ur10e.yml", ...)`
- 使用 `plan_grasp()` 进行 pick/place（approach -> grasp -> lift）
- 末端姿态通过 `GoalToolPose.from_poses()` 指定，`tool_frames=["tool0", "tool1"]`
- 每次规划前调用 `planner.update_world(scene)` 更新碰撞场景

---

## 4. 运行方式

```bash
cd /home/jerry/projects/curobo
python -m lego_stacking.main
# 或
python lego_stacking/main.py
```

浏览器打开 `http://localhost:8080`，通过 Viser GUI 的 "Next Step" 按钮逐步执行堆叠。

---

## 5. 注意事项

- 双臂机器人配置 `dual_ur10e.yml` 中 `tool_frames: ["tool1", "tool0"]`。
- `plan_grasp` 的 `grasp_approach_offset` / `grasp_lift_offset` 为负值表示沿工具 z 轴向下/向上（参考 `simple_stacking_v2.py`）。
- 砖块被抓取时，`held_cube` 的视觉位置需通过 FK 实时更新（参考 `_update_held_cube`）。
- 所有长度单位使用 **米（m）**。
- **所有 lego brick 与机器人之间忽略碰撞检测**，简化场景管理和 pick/place 逻辑。
