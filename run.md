# cuRobo 双臂避障 — 配置与运行指南

## 环境信息

| 项目 | 值 |
|------|-----|
| 源码 | https://github.com/jichuan-yu/curobo |
| 本地路径 | `/home/sicheng/sicheng_srt/curobo_ws` |
| Python 环境 | `.venv/` (虚拟环境，与 ROS2 完全隔离) |
| GPU | RTX 4090 / CUDA 13.2 |
| Torch | 2.9.1 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `dual_arm_viz.py` | **主入口** — 双臂可视化 + 避障规划 |
| `gen_scene_1.py` | 场景 1 生成器（参数化，支持 RL 扫参） |
| `myscene_1.yml` | 场景 1：长条物件搬运避障 |
| `my_scene.yml` | 原始测试场景（7 个障碍物） |
| `test_dual_panda.py` | 双臂配置测试脚本 |
| `curobo/content/configs/robot/dual_franka.yml` | 双臂 Franka 配置 |
| `curobo/content/configs/robot/dual_franka_with_camera.yml` | 双臂+腕部相机配置 (**默认**) |
| `curobo/content/assets/robot/franka_description/dual_franka.urdf` | 双臂 URDF |
| `curobo/content/assets/robot/franka_description/dual_franka_with_camera.urdf` | 双臂+腕部相机 URDF |

**学长分支独有功能：**
| 功能 | 文件 | 用途 |
|------|------|------|
| RelativePoseCost | `curobo/_src/cost/cost_relative_pose.py` | 双臂协同保持相对位姿 |
| arm_relative_pose.py | `dual_panda_benchmark/` | 协调模式示例 |
| lego_stacking | `lego_stacking/` | 完整抓取+装配示例 |

## 启动双臂网页可视化

```bash
cd /home/sicheng/sicheng_srt/curobo_ws
source .venv/bin/activate

# 默认启动场景 1 (长条物件搬运)
python dual_arm_viz.py

# 指定场景
python dual_arm_viz.py --scene myscene_1.yml

# 原始测试场景
python dual_arm_viz.py --scene my_scene.yml

# 指定端口
python dual_arm_viz.py --port 8084
```

浏览器访问 `http://localhost:8083`（VS Code Remote：`Ctrl+Shift+P` → `Forward a Port` → `8083`）。

## 网页操作

- 拖拽绿色控制框设置目标位姿
- 下拉菜单选择 **left / right / both** 规划哪条臂
- 点击 **Plan & Execute** 开始避障规划
- 场景障碍物自动从 YAML 配置文件加载

## 场景管理

### 场景 1：长条物件搬运避障 (`myscene_1.yml`)

```
俯视图:
    Y↑
     │  pillar_p2  pillar_p1  pillar_0  pillar_n1  pillar_n2
     │  [0.30,0.60] [0.30,0.30] [0.30,0.00] [0.30,-0.30] [0.30,-0.60]
     │  (5根高柱, 10cm×10cm×40cm, 间距30cm)
     │
     │  ┌──────────────────────────────────────┐
     │  │ long_part  [1.10, 0.00]             │ ← 200×7×7cm 零件
     │  └──────────────────────────────────────┘
     │
     └──────────────────────────────────────→ X
       0    0.1   0.3                           2.1
            ↑     ↑
         零件近边  高柱排
```

### 参数化生成

```bash
cd /home/sicheng/sicheng_srt/curobo_ws
source .venv/bin/activate

# 默认参数生成
python gen_scene_1.py -o myscene_1.yml

# 自定义参数 (RL 扫参)
python gen_scene_1.py \
    --long_length 2.5 \
    --long_width 0.08 \
    --pillar_count 7 \
    --pillar_spacing 0.25 \
    --pillar_height 0.5 \
    -o myscene_1.yml
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `long_length` | 2.00 | 零件长度 (m) |
| `long_width` | 0.07 | 零件宽度 (m) |
| `long_height` | 0.07 | 零件高度 (m) |
| `long_x_near` | 0.10 | 零件近边距机器人原点 (m) |
| `pillar_x` | 0.10 | 柱子 X 尺寸 (m) |
| `pillar_y` | 0.10 | 柱子 Y 尺寸 (m) |
| `pillar_height` | 0.40 | 柱子高度 (m) |
| `pillar_x_pos` | 0.30 | 柱子排 X 坐标 (m) |
| `pillar_spacing` | 0.30 | 柱子 Y 间距 (m) |
| `pillar_start_y` | -0.60 | 第一根柱子 Y 坐标 (m) |
| `pillar_count` | 5 | 柱子数量 |

## 自定义场景格式 (YAML)

```yaml
# 立方体: [x, y, z, qw, qx, qy, qz]，单位米
cuboid:
  name:
    dims: [dx, dy, dz]
    pose: [x, y, z, qw, qx, qy, qz]

# 球体
sphere:
  name:
    radius: r
    pose: [x, y, z, qw, qx, qy, qz]

# 网面 (STL/OBJ)
mesh:
  name:
    pose: [x, y, z, qw, qx, qy, qz]
    file_path: /path/to/mesh.stl
```

> **注意**：创建新场景后，需将 YAML 复制到 `curobo/content/configs/scene/` 目录，或直接运行 `gen_scene_1.py -o` 自动同步。

## URDF 结构

```
world
├── left_panda_fixed (xyz="0 0.26 0")    # 左臂底座 Y=+0.26
│   └── left_panda_link0 → ... → left_panda_link8
│       └── left_panda_hand + left_panda_camera
│           ├── left_panda_leftfinger    (prismatic, 0~0.04m)
│           ├── left_panda_rightfinger   (prismatic, mimic)
│           └── left_ee_link             (tool frame)
└── right_panda_fixed (xyz="0 -0.26 0")   # 右臂底座 Y=-0.26
    └── right_panda_link0 → ... → right_panda_link8
        └── right_panda_hand + right_panda_camera
            ├── right_panda_leftfinger   (prismatic)
            ├── right_panda_rightfinger  (prismatic, mimic)
            └── right_ee_link            (tool frame)
```

## 运行测试

```bash
cd /home/sicheng/sicheng_srt/curobo_ws
source .venv/bin/activate
python test_dual_panda.py test
```

## 安装记录

```bash
git clone https://github.com/jichuan-yu/curobo.git /home/sicheng/sicheng_srt/curobo_ws
cd /home/sicheng/sicheng_srt/curobo_ws

# 修复 git 代理端口 (7897 → 7890)
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# 创建虚拟环境并安装
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -e ".[cu13-torch]"
```

## 单臂示例 (原始 cuRobo)

```bash
source .venv/bin/activate
python -m curobo.examples.getting_started.motion_planning --visualize
```
