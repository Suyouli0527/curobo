# 双臂末端执行器相对位姿成本函数

## 概述

`RelativePoseCost` 是一个专为双臂机器人设计的成本函数，用于维持两个末端执行器之间的固定相对位姿（相对位置和方向）。这在双臂协作操作任务中非常有用，比如：

- **双臂抓取**：两个机械手抓住同一物体
- **对称操作**：要求两个末端执行器保持固定的相对位置和姿态
- **装配任务**：两个臂需要将两个零件对齐并装配

## 核心概念

### 相对位姿的定义

相对位姿是从**主臂（primary）**到**副臂（secondary）**的相对变换：

```
相对位置 = R_primary^T * (p_secondary - p_primary)
相对四元数 = q_primary^* ⊗ q_secondary
```

其中：
- `p_primary`, `p_secondary` 是两个末端执行器在世界坐标系中的位置
- `q_primary`, `q_secondary` 是两个末端执行器在世界坐标系中的方向（四元数）
- `R_primary^T` 是主臂旋转矩阵的转置
- `q_primary^*` 是主臂四元数的共轭
- `⊗` 表示四元数乘法

## 使用方法

### 1. 配置相对位姿成本

```python
from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg
from curobo._src.types.device_cfg import DeviceCfg
import torch

# 创建配置
device_cfg = DeviceCfg()

relative_pose_cfg = RelativePoseCostCfg(
    device_cfg=device_cfg,
    weight=torch.tensor([1.0]),
    
    # 指定两个末端执行器
    primary_tool_frame="tool0",      # 左臂
    secondary_tool_frame="tool1",    # 右臂
    
    # 目标相对位置 (米): [x, y, z]
    # 表示右臂相对于左臂应该在的位置
    target_relative_position=[0.3, 0.0, -0.1],
    
    # 目标相对方向 (四元数): [w, x, y, z]
    # 这里设置为恒等四元数，表示两臂应该有相同的方向
    target_relative_quaternion=[1.0, 0.0, 0.0, 0.0],
    
    # 成本权重
    position_weight=1.0,       # 位置误差权重
    orientation_weight=0.5,    # 方向误差权重
)
```

### 2. 创建成本函数实例

```python
from curobo._src.cost.cost_relative_pose import RelativePoseCost

relative_pose_cost = RelativePoseCost(relative_pose_cfg)

# 设置批大小和规划地平线
batch_size = 4
horizon = 10
relative_pose_cost.setup_batch_tensors(batch_size, horizon)
```

### 3. 计算成本

```python
# 获取当前末端执行器的位姿
# 这通常来自机器人的正向运动学求解
current_tool_poses = ...  # ToolPose 对象

# 计算成本
cost = relative_pose_cost(current_tool_poses)

# 成本张量形状: [batch_size, horizon, 1]
```

## 参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| `primary_tool_frame` | str | 主臂的末端执行器名称（例如 "tool0"） |
| `secondary_tool_frame` | str | 副臂的末端执行器名称（例如 "tool1"） |
| `target_relative_position` | [x, y, z] | 目标相对位置，单位为米 |
| `target_relative_quaternion` | [w, x, y, z] | 目标相对方向（四元数格式） |
| `position_weight` | float | 位置误差的权重（默认 1.0） |
| `orientation_weight` | float | 方向误差的权重（默认 1.0） |
| `weight` | torch.Tensor | 整体成本函数的权重 |

## 目标相对姿态的设定

### 示例 1：两臂相同方向，距离为 30cm

```python
target_relative_position=[0.3, 0.0, 0.0],      # 30cm in X
target_relative_quaternion=[1.0, 0.0, 0.0, 0.0]  # Identity
```

### 示例 2：两臂相对 180° 旋转

```python
# 绕Z轴旋转180°的四元数是 [0, 0, 0, 1]
target_relative_quaternion=[0.0, 0.0, 0.0, 1.0]
```

### 示例 3：两臂相对 90° 旋转（绕X轴）

```python
# 绕X轴旋转90°：w = cos(45°) = 0.707, x = sin(45°) = 0.707
target_relative_quaternion=[0.707, 0.707, 0.0, 0.0]
```

## 集成到运动规划器

要将 `RelativePoseCost` 集成到 cuRobo 的运动规划器中，需要修改成本管理器配置。

### 步骤 1：修改 `cost_manager_robot_cfg.py`

在 `curobo/_src/rollout/cost_manager/cost_manager_robot_cfg.py` 中添加：

```python
from curobo._src.cost.cost_relative_pose import RelativePoseCost
from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg

# 在 RobotCostManagerCfg 中添加字段
@dataclass
class RobotCostManagerCfg(BaseCostManagerCfg):
    """..."""
    relative_pose_cost_cfg: Optional[RelativePoseCostCfg] = None
```

### 步骤 2：在成本管理器中注册

在 `curobo/_src/rollout/cost_manager/cost_manager_robot.py` 中：

```python
def __init__(self, config: RobotCostManagerCfg):
    """..."""
    # 其他成本函数...
    
    # 添加相对位姿成本
    if config.relative_pose_cost_cfg is not None:
        self.relative_pose_cost = RelativePoseCost(config.relative_pose_cost_cfg)
        self.costs.append(self.relative_pose_cost)
```

## 使用示例

完整的使用示例见：

```bash
python -m curobo.examples.guides.dual_arm_relative_pose
```

## 数学细节

### 四元数操作

1. **四元数乘法**：两个四元数 q1 = [w1, x1, y1, z1] 和 q2 = [w2, x2, y2, z2] 的乘积为：

```
q1 ⊗ q2 = [
    w1*w2 - (x1*x2 + y1*y2 + z1*z2),
    w1*x2 + x1*w2 + (y1*z2 - z1*y2),
    w1*y2 + y1*w2 + (z1*x2 - x1*z2),
    w1*z2 + z1*w2 + (x1*y2 - y1*x2)
]
```

2. **四元数共轭**：q* = [w, -x, -y, -z]

3. **四元数测地距离**：两个单位四元数之间的测地距离为 2*arccos(|q1·q2|)

### 相对位置计算

世界坐标系中的两点相对位置转换到主臂的坐标系中：

```
r_rel = R_primary^T * r_world
```

其中 R_primary 由主臂的四元数 q_primary 定义。

## API 参考

### RelativePoseCost 类

```python
class RelativePoseCost(BaseCost):
    def __init__(self, config: RelativePoseCostCfg)
    
    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs)
        # 设置批处理张量的大小
    
    def forward(self, current_tool_poses: ToolPose, **kwargs) -> torch.Tensor
        # 计算相对位姿成本
        # 返回形状为 [batch_size, horizon, 1] 的成本张量
    
    def position_error(self) -> torch.Tensor
        # 获取最新的位置误差
    
    def orientation_error(self) -> torch.Tensor
        # 获取最新的方向误差
```

### RelativePoseCostCfg 类

```python
@dataclass
class RelativePoseCostCfg(BaseCostCfg):
    primary_tool_frame: str              # 主臂末端执行器名称
    secondary_tool_frame: str            # 副臂末端执行器名称
    target_relative_position: List[float]      # 目标相对位置 [x, y, z]
    target_relative_quaternion: List[float]    # 目标相对方向 [w, x, y, z]
    position_weight: float = 1.0         # 位置误差权重
    orientation_weight: float = 1.0      # 方向误差权重
    rotation_method: str = "quat"        # 旋转表示方法
```

## 常见问题

### Q1: 如何指定两个不同的相对方向？

使用对应的四元数表示。例如，90° 旋转：

```python
# 绕Z轴旋转90°
q_90z = torch.tensor([0.9239, 0.0, 0.0, 0.3827])  # cos(45°), sin(45°)
```

### Q2: 位置权重和方向权重如何平衡？

通过调整 `position_weight` 和 `orientation_weight` 参数。例如：

- 优先保持相对位置：`position_weight=1.0, orientation_weight=0.1`
- 优先保持相对方向：`position_weight=0.1, orientation_weight=1.0`

### Q3: 可以否定相对位姿的某些轴吗？

当前实现没有轴选择功能。如果需要，可以修改成本函数或在规划器级别添加约束。

## 参考文献

- Quaternion Mathematics: https://en.wikipedia.org/wiki/Quaternion
- Robot Kinematics and Dynamics: Craig, J. J. (2004). Introduction to Robotics: Mechanics and Control.
