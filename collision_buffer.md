# 碰撞 Buffer 配置

> 来源: `curobo/content/configs/robot/dual_franka_with_camera.yml`

## 全局参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `collision_sphere_buffer` | 0.03 | 全局统一 buffer，对齐学长 XACRO `safety_distance=0.03` |

## 自碰撞 Buffer (`self_collision_buffer`)

加到每个 sphere 半径上的额外 padding。单位: 米。

```
所有 link:                 0.00  (全局 collision_sphere_buffer=0.03 统一提供)
```

| Link | Buffer (m) | 有效总 padding (含全局 0.03) |
|------|-----------|------|
| 全部 link | 0.00 | 0.03 |

### 自碰撞代价中的使用

CUDA kernel 中每对 sphere pair $(i,j)$ 的碰撞判定：

$$f_{\text{diff}} = (r_i + b_i + r_j + b_j)^2 - \|\mathbf{c}_i - \mathbf{c}_j\|^2$$

若 $f_{\text{diff}} > 0$，代价 $\displaystyle C = w_{\text{self}} \cdot \frac{1}{2} \cdot f_{\text{diff}}$。

buffer $b$ 使某些 link 在检测中"变胖"，提前预警而非等真正穿透。

## 与学长模型的对齐

| 学长的 XACRO | 我们的 YAML | 说明 |
|---|---|---|
| `safety_distance=0.03` | `collision_sphere_buffer: 0.03` | 全局统一加在所有 sphere 半径上 |
| 无 per-link buffer | `self_collision_buffer: {link0: 0.02, link1: 0.02}` | 仅底座和肩部保留额外缓冲 |

有效碰撞半径 = `sphere_radius + 0.03(global) + self_collision_buffer[link]`

## 场景碰撞激活距离

| 参数 | 值 | 说明 |
|------|-----|------|
| `activation_distance` | 0.03 m | 球体膨胀距离，用于平滑激活函数 |

穿透深度 $d^{\text{pen}} = r + 0.03 - \operatorname{sdf}(\mathbf{c})$，过平滑激活 $f_\eta(d)$ 后乘权重。

## 碰撞球体 (`collision_spheres`)

每个 link 用多个 sphere 近似几何形状（双臂 L+R 完全相同）：

| Link | 球数 | 典型半径 (m) | 说明 |
|------|------|-------------|------|
| link0 (base) | 2 | 0.030 | 底座 |
| link1 (shoulder) | 4 | 0.055~0.060 | 肩部，粗杆 |
| link2 | 4 | 0.055 | 上臂 |
| link3 | 4 | 0.050~0.060 | 肘部 |
| link4 | 4 | 0.052~0.055 | 前臂 |
| link5 | 13 | 0.022~0.050 | 腕部，球最多、最小 |
| link6 | 3 | 0.045 | 腕旋转 |
| link7 | 5 | 0.020~0.045 | 末端法兰 |
| hand | 18 | 0.022~0.023 | 手掌，球最多 |
| leftfinger | 2 | 0.011 | 手指，球最小 |
| rightfinger | 2 | 0.011 | 手指 |
| wrist_camera | 4 | 0.017~0.018 | 相机（mesh 用，无碰撞检测） |
| attached_object (预留) | 32×2 | 运行时 | 抓取物体 sphere 拟合 |

**单臂总球数**: $2+4+4+4+4+13+3+5+18+2+2+4 = 65$

**双臂总球数**: $65 \times 2 = 130$（抓取启用时 $+64 = 194$）

## 碰撞 Link 列表 (`collision_link_names`)

参与碰撞检测的 link（共 26 个，含 `attached_object`）：

```
panda_right_link0~7, wrist_camera, hand, leftfinger, rightfinger,
attached_object_right,
panda_left_link0~7,  wrist_camera, hand, leftfinger, rightfinger,
attached_object_left
```

## 自碰撞忽略 (`self_collision_ignore`)

### 忽略规则

- **同 link 内不检测**（link 自身的 sphere 间不互检）
- **相邻 link 不检测**（运动学上不可能自碰，如 link3↔link4）
- **手指只忽略与相邻手掌间的碰撞**（finger↔hand），不忽略 finger↔finger（双手指尖可能相碰）

### 跨臂忽略

双臂仅底座和肩部互相忽略（距离远，不可能碰撞）：

```
panda_right_link0 ↔ panda_left_link0~3
panda_left_link0  ↔ panda_right_link1~3
panda_right_link1 ↔ panda_left_link1~2,4
panda_left_link1  ↔ panda_right_link2,4
```

末端（link3~7, hand, finger）之间**不忽略**——双手操作时指尖可能相碰。

## 运行时抓取

`extra_collision_spheres` 为 `attached_object_left/right` 各预留 32 个 sphere 槽位。抓取时 `AttachmentManager.fit_spheres()` 采样物体表面填充。

## 场景碰撞缓存

```yaml
collision_cache:
  cuboid: 50
  mesh: 10
```

预分配 GPU buffer：最多 50 个 cuboid + 10 个 mesh 障碍物。
