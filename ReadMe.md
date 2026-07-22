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

### 优化问题

$$\min_{\mathbf{u} \in \mathbb{R}^{N \times d}} \quad \mathcal{L}(\mathbf{u}) = J_{\text{pose}} + J_{\text{bound}} + J_{\text{reg}} + J_{\text{target}} + C_{\text{scene}} + C_{\text{self}} + J_{\text{rel}}$$

其中 $N=16$ 为 B-spline knot 数 (action horizon)，$d=14$ 为关节自由度 (双臂各 7 DoF)。

所有代价项均为**软代价**（连续、可微），优化器（LBFGS）直接最小化其加权和。碰撞等项在优化中使用软代价形式，在指标评估（metrics rollout）中通过 `convert_to_binary` 切换为二值可行性判定。

---

### 1. 末端位姿追踪 $J_{\text{pose}}$

对每个末端执行器 frame $f \in \mathcal{F}$，使用 **axis-angle** 方法（`use_lie_group: false`）：

$$J_{\text{pose}} = \sum_{f \in \mathcal{F}} \sum_{k=1}^{N} \left[ \frac{w_p}{2} \bigl\|\mathbf{W}_p \odot (\mathbf{p}_{k,f} - \mathbf{p}_f^{\text{goal}})\bigr\|^2 + w_r \|\boldsymbol{\omega}_{k,f}\|^2 \right]$$

其中 $\boldsymbol{\omega}_{k,f} = \theta_{k,f} \cdot \mathbf{a}_{k,f}$ 为 axis-angle 表示，由 $\mathbf{q}_{k,f} \otimes (\mathbf{q}_f^{\text{goal}})^{-1}$ 提取：
$\theta = 2\,\mathrm{arctan2}(\|\mathbf{v}\|, |w|)$，$\mathbf{a} = \mathbf{v}/\|\mathbf{v}\|$，$\mathbf{v} = \mathbf{W}_r \odot \text{vec}(\mathbf{q}_\Delta)$。

位置梯度：$\nabla_{\mathbf{p}} = w_p \cdot \mathbf{W}_p^2 \odot (\mathbf{p} - \mathbf{p}^{\text{goal}})$。

姿态梯度：$\nabla_{\boldsymbol{\omega}} = 2 w_r \boldsymbol{\omega}$（当 $w \ge 0$）。

> 注：当 `use_lie_group: true` 时，改用 Lie 群对数映射 $\log: SO(3) \to \mathfrak{so}(3)$，公式等价但 Jacobian 计算更精确。

---

### 2. 关节空间 Bound 惩罚 $J_{\text{bound}}$

对每个关节 $d$ 的每个状态量 $s \in \{\text{pos}, \text{vel}, \text{acc}, \text{jerk}, \text{torque}\}$，
当状态超出**激活距离收缩后的边界**时施加二次惩罚：

$$J_{\text{bound}} = \sum_{k=1}^{N} \sum_{d=1}^{d} \sum_{s} \frac{w_b^s}{2} \Big( \bigl[\max(0,\; x_{k,d}^s - \bar{u}_d^s)\bigr]^2 + \bigl[\max(0,\; \bar{l}_d^s - x_{k,d}^s)\bigr]^2 \Big)$$

其中收缩边界为：
$$\bar{l}_d^s = l_d^s + \eta^s (u_d^s - l_d^s), \qquad \bar{u}_d^s = u_d^s - \eta^s (u_d^s - l_d^s)$$

| 状态 $s$ | 边界权重 $w_b^s$ | 激活距离 $\eta^s$ |
|----------|-----------------|-------------------|
| position | 1000 | 0.01 |
| velocity | 1000 | 0.01 |
| acceleration | 1000 | 0.01 |
| jerk | 100 | 0.01 |
| torque | 0 | 0.01 |

梯度：当 $x > \bar{u}$ 时 $\nabla = w_b^s (x - \bar{u})$；当 $x < \bar{l}$ 时 $\nabla = w_b^s (x - \bar{l})$；否则 $\nabla = 0$。

`retime_weights: false` 表示权重不随 $dt$ 缩放。

---

### 3. L2 平滑正则化 $J_{\text{reg}}$

对速度、加速度、加加速度始终施加 Tikhonov 正则化（与 bound 惩罚独立）：

$$J_{\text{reg}} = \sum_{k=1}^{N} \sum_{d=1}^{d} \frac{1}{2} \Big[ \lambda_v \dot{q}_{k,d}^2 + \lambda_a \ddot{q}_{k,d}^2 + \lambda_j \dddot{q}_{k,d}^2 \Big]$$

| 正则化项 | 权重 | 说明 |
|---------|------|------|
| $\lambda_v$ (velocity) | 0.01 | 微弱阻尼 |
| $\lambda_a$ (acceleration) | 10000 | **主导平滑项**，抑制急加速 |
| $\lambda_j$ (jerk) | 10 | 抑制加加速度突变 |
| torque smooth | 0 | 未启用 |
| energy | 0 | 未启用 |

`retime_regularization_weights: true` 表示权重按 $dt^n$ 缩放（如 $\lambda_a$ 缩放 $dt^2$），使正则化近似时间积分，对不同 $dt$ 保持一致效果。

---

### 4. 关节目标距离 $J_{\text{target}}$

将轨迹拉向默认关节配置（retract pose），终端权重远大于非终端：

$$J_{\text{target}} = \sum_{k=1}^{N} \alpha_k \, w_t \sum_{d=1}^{d} (q_{k,d} - q_d^{\text{target}})^2$$

$$\alpha_k = \begin{cases} 1.0 & k = N \text{ (terminal knot)} \\ 0.05 & k < N \text{ (non-terminal)} \end{cases}$$

梯度：$\nabla_{q_{k,d}} = 2 \alpha_k w_t (q_{k,d} - q_d^{\text{target}})$。

---

### 5. 相对位姿协同 $J_{\text{rel}}$（运行时可选）

简单 L2 代价，始终可微，LBFGS 最擅长的形式。

$$J_{\text{rel}} = \sum_{k=1}^{N} \Big[ \frac{w_{\text{rel},p}}{2} \|\Delta\mathbf{p}_k - \Delta\mathbf{p}^{\text{target}}\|^2 + w_{\text{rel},r} \|\boldsymbol{\omega}_k^{\text{rel}}\|^2 \Big]$$

其中相对位姿在 primary 局部坐标系中计算：
$$\Delta\mathbf{p}_k = \mathbf{q}_{k,\text{primary}}^* \odot (\mathbf{p}_{k,\text{secondary}} - \mathbf{p}_{k,\text{primary}})$$
$$\Delta\mathbf{q}_k = \mathbf{q}_{k,\text{primary}}^* \otimes \mathbf{q}_{k,\text{secondary}}$$

---

### 6. 场景碰撞代价 $C_{\text{scene}}$

对每个 robot sphere $i$ 在每个时间步，使用 **平滑激活函数**（C¹ 连续，quadratic-linear transition）：

$$C_{\text{scene}} = w_{\text{col}} \sum_{k=1}^{N} \sum_{i=1}^{N_s} f_\eta\bigl(d_{k,i}^{\text{pen}}\bigr)$$

其中穿透深度为 $d_{k,i}^{\text{pen}} = r_i + a_{\text{col}} - \text{sdf}(\mathbf{c}_i)$（sdf 在障碍物内部为负），
平滑激活函数定义（`wp_collision_common.py:12-38`）：

$$f_\eta(d) = \begin{cases}
0 & d \le 0 \\
\dfrac{d^2}{2\eta} & 0 < d \le \eta \quad \text{(quadratic region)} \\
d - \dfrac{\eta}{2} & d > \eta \quad \text{(linear region)}
\end{cases}$$

其中 $\eta = a_{\text{col}} = 0.03$ m。该函数在 $d=0$ 和 $d=\eta$ 处 C¹ 连续，梯度分别为 $0$（d=0 时）、$d/\eta$（中间）、$1$（线性区）。

Sweep 模式 (`use_sweep: true`, `use_speed_metric: true`) 额外考虑 sphere 在 $dt$ 内的运动速度，动态缩放激活距离。

---

### 7. 自碰撞代价 $C_{\text{self}}$

对所有预定义的 sphere pair $(i,j)$，使用平方距离差惩罚：

$$C_{\text{self}} = w_{\text{self}} \sum_{k=1}^{N} \max_{(i,j)} \left[ \frac{1}{2} \bigl[\max(0,\; (r_i + r_j + \delta_{ij})^2 - \|\mathbf{c}_{k,i} - \mathbf{c}_{k,j}\|^2)\bigr] \right]$$

其中 $\delta_{ij}$ 为 pair-specific padding（碰撞缓冲）。梯度：

$$\nabla_{\mathbf{c}_i} = -w_{\text{self}} (\mathbf{c}_i - \mathbf{c}_j), \quad \nabla_{\mathbf{c}_j} = w_{\text{self}} (\mathbf{c}_i - \mathbf{c}_j)$$

仅在最大穿透 pair 上计算梯度（稀疏梯度）。

---

### 权重总表

| 符号 | 含义 | 源 YAML / 代码 | 值 |
|------|------|-------------|-----|
| $w_p$ | 末端位置追踪 | `tool_pose_cfg.weight[0]` | 50000 |
| $w_r$ | 末端姿态追踪 | `tool_pose_cfg.weight[1]` | 200 |
| $w_b^{\text{pos}}$ | 关节位置 bound | `cspace_cfg.weight[0]` | 1000 |
| $w_b^{\text{vel}}$ | 关节速度 bound | `cspace_cfg.weight[1]` | 1000 |
| $w_b^{\text{acc}}$ | 关节加速度 bound | `cspace_cfg.weight[2]` | 1000 |
| $w_b^{\text{jerk}}$ | 关节 jerk bound | `cspace_cfg.weight[3]` | 100 |
| $\lambda_v$ | 速度 L2 正则化 | `squared_l2_regularization_weight[0]` | 0.01 |
| $\lambda_a$ | 加速度 L2 正则化 | `squared_l2_regularization_weight[1]` | 10000 |
| $\lambda_j$ | Jerk L2 正则化 | `squared_l2_regularization_weight[2]` | 10 |
| $\eta^s$ | Bound 激活距离（全部 5 维） | `activation_distance[*]` | 0.01 |
| $w_t$ | 关节目标距离 | `cspace_target_weight` | 1000 |
| $\alpha_k$ | 终端权重因子 | `cspace_non_terminal_weight_factor` | 1 (term), 0.05 (non-term) |
| $w_{\text{rel},p}$ | 协同位置 | `controller.py` | 7500 |
| $w_{\text{rel},r}$ | 协同姿态 | `controller.py` | 750 |
| $w_{\text{col}}$ | 场景碰撞软代价 | `scene_collision_cfg.weight` | 10000 |
| $a_{\text{col}}$ | 碰撞激活距离 $\eta$ | `scene_collision_cfg.activation_distance` | 0.03 m |
| $w_{\text{self}}$ | 自碰撞软代价 | `self_collision_cfg.weight` | 100000 |

### 优化器参数

| 参数 | 值 |
|------|-----|
| $N$ (`n_knots`) | 16 |
| $\Delta t$ (`optimization_dt`) | 0.025 s |
| prediction horizon | $N \cdot \Delta t = 0.4$ s |
| cold start iters | 100 |
| warm start iters | 200 |
| LBFGS `num_iters` | 40 |
| LBFGS `inner_iters` | 20 |
| `step_scale` | 0.8 |
| `history` | 27 |
