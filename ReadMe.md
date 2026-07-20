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

## MPC 目标函数

### 软代价

$$\min_{u} \sum_{k=1}^{N} J_{\text{pose}} + J_{\text{cspace}} + J_{\text{reg}} + J_{\text{target}} + J_{\text{rel}}$$

**末端追踪：**

$$J_{\text{pose}} = \frac{50000}{2} \|\mathbf{p}_k - \mathbf{p}_{goal}\|^2 + 200 \| \log(\mathbf{q}_k^{-1} \otimes \mathbf{q}_{goal}) \|^2$$

**关节空间：**

$$J_{\text{cspace}} = 1000 \|\dot{\mathbf{q}}\|^2 + 1000 \|\ddot{\mathbf{q}}\|^2 + 1000 \|\dddot{\mathbf{q}}\|^2 + 100 \|\dot{\boldsymbol{\tau}}\|^2$$

**L2 正则：**

$$J_{\text{reg}} = 0.01 \|\dot{\mathbf{q}}\|^2 + 10000 \|\ddot{\mathbf{q}}\|^2 + 10 \|\dddot{\mathbf{q}}\|^2$$

**关节目标：**

$$J_{\text{target}} = \alpha_k \cdot 1000 \cdot \|\mathbf{q}_k - \mathbf{q}_{target}\|^2, \quad \alpha_k = \begin{cases} 1, & k=N \\ 0.05, & k<N \end{cases}$$

**双臂协同：**

$$J_{\text{rel}} = \frac{50000}{2} \|\Delta\mathbf{p} - \Delta\mathbf{p}_{target}\|^2 + 5000 \| \log(\Delta\mathbf{q}^{-1} \otimes \Delta\mathbf{q}_{target}) \|^2$$

### 硬约束

**场景碰撞 (swept-sphere, act=30mm)：**

$$C_{\text{scene}} = 10000 \cdot \sum_i \max(0,\ 0.03 - d_i) \leq 0$$

**自碰撞：**

$$C_{\text{self}} = 100000 \cdot \sum_{(i,j)} \max(0,\ pad_{ij} - d_{ij}) \leq 0$$

**双臂协同 (tol 10mm / 0.05rad)：**

$$C_{\text{rel}} = \begin{cases} 0, & \|\Delta\mathbf{p} - \Delta\mathbf{p}_{target}\| < 0.01 \land \theta < 0.05 \\ 500000 \cdot e_p + 50000 \cdot e_r, & \text{否则} \end{cases}$$

### 预测参数

| 参数 | 值 |
|------|-----|
| `n_knots` (action horizon) | 16 |
| `optimization_dt` | 0.025s |
| `interpolation_steps` | 4 |
| prediction horizon | 0.4s |
| cold start | 100 iters |
| warm start | 100 iters |
| LBFGS num_iters | 40 |
| LBFGS inner_iters | 20 |
| step_scale | 0.8 |
