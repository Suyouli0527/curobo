# 双臂 Panda 避障协同搬运

基于 cuRobo MPC 的双臂 Franka Panda 实时避障与协同搬运系统。

## 环境准备

```bash
cd ~/csc_curobo
source .venv/bin/activate
```

## 启动

```bash
# 默认场景 1
python -m run.main

# 指定场景
python -m run.main --scene myscene_2.yml

# 自定义端口
python -m run.main --scene myscene_2.yml --port 8081
```

### 可用场景 (`run/myScene/`)

| 场景文件 | 说明 |
|---------|------|
| `myscene_1.yml` | 基础场景 |
| `myscene_2.yml` | 柜子 + 长条物体（双臂搬运） |
| `myscene_3.yml` | — |
| `myscene_5.yml` | — |
| `my_scene.yml` | — |

## 操作

打开浏览器访问 `http://localhost:8080`：

| 操作 | 方式 | 说明 |
|------|------|------|
| 设定目标 | 拖拽红/蓝控制框 | MPC 实时追踪目标位姿 |
| 协同运动 | 点击 **Coordinate** | 双臂相对位姿锁定，拖一臂另一臂跟随 |
| 抓取物体 | 点击 **Grasp** | 自动抓取离双手末端中心最近的障碍物 |
| 释放物体 | 点击 **Release** | 释放已抓取物体 |

## 项目结构

```
run/
├── main.py          # 入口脚本
├── controller.py    # MPC 控制器 (初始化/协同/抓取/步进)
├── gripper.py       # 抓取模块 (障碍物选取/可视化)
├── scene.py         # 场景加载与 Viser 渲染
├── recorder.py      # 数据记录与画曲线
├── utils.py         # 位姿/四元数工具函数
└── myScene/         # 场景 YAML 文件
```

## 退出

按 `Ctrl+C` 退出，自动保存性能曲线到 `run/plots/`。
