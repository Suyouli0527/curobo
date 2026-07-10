# 科研日志 — 2026-07-07

## 今日总览

工作时间：8:00 ~ 22:00+  
主要目标：cuRobo 双臂 Panda 避障规划（场景搭建 + 响应式 MPC 控制 + 双臂协同运动）

---

## 上午：场景搭建

- 创建了 5 个测试场景配置文件（myscene_1~5.yml），覆盖长条物件搬运、多层货架拣选、椅子装配等任务场景
- 编写了 gen_scene_1.py 和 gen_scene_2.py 两个参数化场景生成器，支持命令行调节障碍物数量、尺寸、间距等参数
- 新增了 3 个机器人配置文件：dual_panda.yml（双臂）、mj_left_arm.yml / mj_right_arm.yml（单臂）
- 补充了 URDF 模型文件和相关 STL 网格
- 修复了 dual_franka_with_camera.urdf 中右臂底座 Y 坐标的错误

## 下午+晚上：MPC 调试

### Reactive MPC 改造
- 参照官方 reactive_control.py，将 dual_arm_viz.py 从手动步进模式改为连续响应式 MPC
- 移除 200 步上限，改为 while True 持续循环，每步 warm-start 重优化
- 去掉"Move"按钮，改为实时检测控制框位姿变化自动追踪

### CUDA 线程安全问题
- **问题**：Viser 按钮回调在子线程中调用 IK 求解（CUDA graph capture），与主线程的 MPC CUDA graph replay 冲突，导致 stream capture invalidated
- **解决**：所有 CUDA 操作延迟到主线程执行。按钮回调仅设置 pending 标志（pending_goal、pending_coord_toggle），主循环在安全时机消费

### 双臂协同运动
- 第一阶段尝试：RelativePoseCost + 双臂位姿检测对齐。效果不理想，ToolPoseCost 与 RelativePoseCost 梯度冲突
- 第二阶段（当前方案）：点击 Coordinate 后将两个独立控制框合并为单一控制框。拖拽合并框时自动推算双臂目标位姿，从根本上保证相对位姿固定
  - 合并框位置 = 两臂末端中点
  - 合并框姿态 = q_left * q_right
  - 记录合并框到各臂的偏移，拖拽时反算

### 设备不匹配修复
- 协同模式下 numpy（控制框读取）与 CUDA tensor（FK 偏移量）混用导致 device mismatch
- 解决方案：偏移量在捕获时转为 numpy，协同目标计算全部在 numpy 中完成，最后一次性转为 CUDA tensor

### 其他杂项
- 修复 venv 硬编码路径（从 /home/sicheng/sicheng_srt/curobo_ws 改为 /home/sicheng/csc_curobo）
- 提交到新分支 csc_srt（未推送，待配 GitHub 认证）

---

## 关键文件

| 文件 | 说明 |
|------|------|
| dual_arm_viz.py | 主入口：双臂 reactive MPC + Viser 可视化 + 协同模式 |
| gen_scene_1.py | 长条物件搬运场景参数化生成器 |
| gen_scene_2.py | 多层货架拣选场景参数化生成器 |
| curobo/content/configs/scene/myscene_1~5.yml | 5 个场景配置文件 |
| curobo/content/configs/robot/dual_panda.yml | 双臂机器人配置 |

## 待办

- [ ] 配置 GitHub token，推送 csc_srt 分支
- [ ] 在真实场景中测试协同搬运效果
- [ ] 清理根目录下重复的 scene yml 文件
