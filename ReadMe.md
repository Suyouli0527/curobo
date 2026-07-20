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

$$\min_{\mathbf{u} \in \mathbb{R}^{N \times d}} \quad \underbrace{J_{\text{pose}} + J_{\text{cspace}} + J_{\text{reg}} + J_{\text{target}} + J_{\text{rel}}}_{\text{软代价 } \mathcal{L}(\mathbf{u})} + \underbrace{C_{\text{scene}} + C_{\text{self}} + C_{\text{rel}}}_{\text{硬约束 } \Phi(\mathbf{u})}$$

其中 $N=16$ 为 action horizon，$d=14$ 为关节自由度。

### 软代价 $\mathcal{L}(\mathbf{u})$

$$\begin{aligned}
J_{\text{pose}} &= \sum_{k=1}^{N} \left[ \frac{w_p}{2} \|\mathbf{p}_k - \mathbf{p}^{\text{goal}}\|^2 + w_r \| \log\!\big(\mathbf{q}_k^{-1} \otimes \mathbf{q}^{\text{goal}}\big) \|^2 \right] \\[4pt]
J_{\text{cspace}} &= \sum_{k=1}^{N} \left[ w_v \|\dot{\mathbf{q}}_k\|^2 + w_a \|\ddot{\mathbf{q}}_k\|^2 + w_j \|\dddot{\mathbf{q}}_k\|^2 \right] \\[4pt]
J_{\text{reg}} &= \sum_{k=1}^{N} \left[ \lambda_v \|\dot{\mathbf{q}}_k\|^2 + \lambda_a \|\ddot{\mathbf{q}}_k\|^2 + \lambda_j \|\dddot{\mathbf{q}}_k\|^2 \right] \\[4pt]
J_{\text{target}} &= \sum_{k=1}^{N} \alpha_k \, w_t \, \|\mathbf{q}_k - \mathbf{q}^{\text{target}}\|^2 \\[4pt]
J_{\text{rel}} &= \sum_{k=1}^{N} \left[ \frac{w_{\text{rel},p}}{2} \|\Delta\mathbf{p}_k - \Delta\mathbf{p}^{\text{target}}\|^2 + w_{\text{rel},r} \| \log\!\big(\Delta\mathbf{q}_k^{-1} \otimes \Delta\mathbf{q}^{\text{target}}\big) \|^2 \right]
\end{aligned}$$

### 硬约束 $\Phi(\mathbf{u})$ (binary / barrier)

$$\begin{aligned}
C_{\text{scene}} &= w_{\text{col}} \sum_{k=1}^{N} \sum_{i=1}^{N_s} \max\!\big(0,\ a_{\text{col}} - d_{k,i}\big) \\[4pt]
C_{\text{self}} &= w_{\text{self}} \sum_{k=1}^{N} \sum_{(i,j)} \max\!\big(0,\ \text{pad}_{ij} - d_{k,ij}\big) \\[4pt]
C_{\text{rel}} &= w_{\text{rel}}^{\text{hard}} \sum_{k=1}^{N} \Big[ \max\!\big(0,\ \|\Delta\mathbf{p}_k - \Delta\mathbf{p}^{\text{target}}\| - \delta_p\big) + \max\!\big(0,\ \theta_k - \delta_r\big) \Big]
\end{aligned}$$

### 权重表

| 符号 | 含义 | 值 |
|------|------|-----|
| $w_p$ | 末端位置追踪 | 50000 |
| $w_r$ | 末端姿态追踪 | 200 |
| $w_v, w_a, w_j$ | 关节速度/加速度/加加速度 | 1000, 1000, 1000 |
| $\lambda_v, \lambda_a, \lambda_j$ | L2 正则化 | 0.01, 10000, 10 |
| $w_t$ | 关节目标距离 | 1000 |
| $\alpha_k$ | 终端权重因子 | 1 (terminal), 0.05 (non-terminal) |
| $w_{\text{rel},p}, w_{\text{rel},r}$ | 协同软约束 (pos/rot) | 50000, 5000 |
| $w_{\text{col}}$ | 场景碰撞权重 | 10000 |
| $a_{\text{col}}$ | 碰撞激活距离 | 0.03 m |
| $w_{\text{self}}$ | 自碰撞权重 | 100000 |
| $w_{\text{rel}}^{\text{hard}}$ | 协同硬约束 (pos/rot) | 500000, 50000 |
| $\delta_p, \delta_r$ | 协同容差 | 0.01 m, 0.05 rad |

### 优化器参数

| 参数 | 值 |
|------|-----|
| $N$ (`n_knots`) | 16 |
| $\Delta t$ (`optimization_dt`) | 0.025 s |
| prediction horizon | $N \cdot \Delta t = 0.4$ s |
| cold start iters | 100 |
| warm start iters | 100 |
| LBFGS `num_iters` | 40 |
| LBFGS `inner_iters` | 20 |
| `step_scale` | 0.8 |
| `history` | 27 |
