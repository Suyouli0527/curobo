# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

cuRobo is a CUDA-accelerated robot motion generation library. It provides GPU-parallel algorithms for forward/inverse kinematics, collision checking, trajectory optimization, geometric planning, and motion generation, scaling from single-arm manipulators to high-DoF humanoids.

This is **cuRoboV2**, a significant rewrite from v1. The public API has changed. If v1 API compatibility is needed, pin to tag `v0.7.8`.

## Development Commands

### Installation

The project uses `uv` for dependency management and `setuptools` for building. CUDA kernels are compiled at runtime via `cuda.core`, not during package installation.

```bash
# Create virtual environment (Python >=3.10)
uv venv --python 3.11
source .venv/bin/activate

# Install with CUDA 13 + PyTorch (check nvidia-smi for CUDA version)
uv pip install -e .[cu13-torch] --no-build-isolation

# Install with dev dependencies
uv pip install -e .[cu13-torch,dev] --no-build-isolation
```

**Important**: If Gurobi's bundled Python (`~/softwares/gurobi*/linux64/bin/python3.11`) appears earlier in `PATH` than the system Python, the virtual environment will use Gurobi's incomplete standard library and fail. Ensure system Python 3.11 is used: `uv venv --python /usr/bin/python3.11`.

### Running Tests

```bash
# Run all tests (parallel with 4 workers, default from pyproject.toml)
pytest --pyargs curobo.tests

# Run a specific test module
pytest curobo/tests/_src/motion/test_motion_planner.py -xvs

# Run a specific test
pytest curobo/tests/_src/robot/kinematics/test_kinematics.py::TestKinematics::test_fk -xvs

# Run on CPU only
pytest curobo/tests/_src/robot/kinematics/test_kinematics.py --no-cuda

# Run with full parameter ranges (parametric tests)
pytest curobo/tests/ --full-params
```

### Linting and Formatting

```bash
# Format and lint (ruff, line-length=99, Google docstring convention)
ruff check curobo/
ruff format curobo/

# Type checking (ty)
ty check curobo/
```

### Running Examples

```bash
# Motion planning
python -m curobo.examples.getting_started.motion_planning

# IK (single, batched, collision-free)
python -m curobo.examples.getting_started.inverse_kinematics

# Interactive Viser viewer
python -m curobo.examples.getting_started.motion_planning --visualize

# Reachability analysis with dual-arm robot
python -m curobo.examples.getting_started.inverse_kinematics --reachability --robot dual_ur10e.yml

# Custom optimization (Rosenbrock with MPPI/ES/L-BFGS/Torch/SciPy)
python -m curobo.examples.guides.custom_optimization
```

## High-Level Architecture

### Public API vs. Internal Implementation

The codebase uses a **thin public API layer** over an internal implementation:

- **Public modules** (`curobo/*.py`): Re-export classes from `_src` with docstrings and examples. These are the intended user-facing surfaces.
  - `curobo.motion_planner` → `curobo._src.motion.motion_planner`
  - `curobo.inverse_kinematics` → `curobo._src.solver.solver_ik`
  - `curobo.scene` → `curobo._src.geom.types`
  - `curobo.optim` → `curobo._src.optim.*`

- **Internal modules** (`curobo/_src/`): The actual implementation. Most development work happens here.

### SolverCore: Shared Infrastructure Pattern

`SolverCore` (`curobo/_src/solver/solver_core.py`) is **not a base class** -- it is a composition component owned by each solver:

```
IKSolver        owns a SolverCore
TrajOptSolver   owns a SolverCore
MPCSolver       owns a SolverCore
```

`SolverCore` manages:
- **Rollouts**: `metrics_rollout`, `optimizer_rollouts`, `auxiliary_rollout`, `additional_metrics_rollouts`
- **Optimizer**: `MultiStageOptimizer` wrapping one or more optimizers
- **Collision**: `SceneCollision` checker (shared across rollouts)
- **Kinematics**: `Kinematics` instance (shared across rollouts)
- **Seed management**: `SeedManager` for generating initial guesses
- **Goal management**: `GoalManager` for batched goal poses
- **Attachment management**: `AttachmentManager` for attaching obstacles to robot links

### RobotRollout: The Optimization Objective

`RobotRollout` (`curobo/_src/rollout/rollout_robot.py`) is the central object that connects the three pillars of optimization:

1. **State Transition Model** (`RobotStateTransition`): Forward-simulates joint trajectories from actions. Supports position clique, B-spline knot, acceleration, and teleport control modes.
2. **Cost Manager** (`RobotCostManager`): Evaluates cost terms (pose tracking, collision, self-collision, joint limits, smoothness).
3. **Collision Checker** (`SceneCollision`): GPU-accelerated distance queries against scene obstacles.

The optimizer (e.g., LBFGS, MPPI) calls `rollout.compute_metrics_from_action(actions)` which returns `RolloutMetrics` containing costs and constraints.

### Configuration Factory Pattern

All major components use a **config dataclass + `create()` factory** pattern:

```python
# Configs accept file paths, dicts, or existing config objects
config = MotionPlannerCfg.create(
    robot="franka.yml",                    # path to YAML in curobo/content/configs/robot/
    scene_model="collision_test.yml",      # path to YAML in curobo/content/configs/scene/
    ik_optimizer_configs=["ik/lbfgs_ik.yml"],
    trajopt_optimizer_configs=["trajopt/lbfgs_bspline_trajopt.yml"],
)
planner = MotionPlanner(config)
```

Config resolution order:
1. `curobo/content/configs/` for built-in configs
2. Absolute paths for user configs
3. Dicts for inline configuration

The `create()` method chains through: `MotionPlannerCfg.create()` → `IKSolverCfg.create()` / `TrajOptSolverCfg.create()` → `SolverCoreCfg.create()` → `RobotRolloutCfg.create_with_component_types()` → individual cost/transition model configs.

### CUDA Graph Management

Many components support `use_cuda_graph=True` for lower kernel-launch overhead. **Critical**: Call `destroy()` on planners/solvers before dropping references to avoid `cudaMallocAsync` warnings:

```python
planner = MotionPlanner(config)
try:
    result = planner.plan_pose(...)
finally:
    planner.destroy()  # Releases cuGraphExec resources
```

### Warmup Protocol

All solvers require a `warmup()` call before real use to pre-compile CUDA graphs and JIT kernels:

```python
planner.warmup(enable_graph=True, num_warmup_iterations=5)
ik_solver.warmup(enable_graph=True)
```

Without warmup, the first solve call will be significantly slower due to JIT compilation.

## Creating Simulation Scenes

Scenes are represented by `SceneCfg` (aliased as `Scene` in the public API) and managed by `SceneCollision`.

### Scene Obstacle Types

Supported obstacle primitives (`curobo/_src/geom/types/`):
- `Cuboid`: Axis-aligned box with `dims=[x, y, z]` and `pose=[x, y, z, qw, qx, qy, qz]`
- `Sphere`: `radius` and `pose`
- `Capsule`: `radius`, `length`, and `pose`
- `Cylinder`: `radius`, `height`, and `pose`
- `Mesh`: `file_path` to `.obj`/`.stl` and `pose`
- `VoxelGrid`: Dense voxel grid for ESDF perception

**Important bug**: `SceneData.from_scene_cfg` (`curobo/_src/geom/data/data_scene.py`) only loads `cuboid`, `mesh`, and `voxel` into the GPU collision buffer. `Cylinder`, `Sphere`, and `Capsule` obstacles defined in a `SceneCfg` are silently ignored by the collision checker, even though they are valid `SceneCfg` types and render correctly in Viser. To use cylindrical obstacles for collision checking, approximate them with `Cuboid` (bounding box) or convert the entire scene to meshes via `SceneCfg.create_mesh_scene(scene)` before passing to the planner.

### Programmatic Scene Construction

```python
from curobo.scene import Scene, Cuboid, Sphere, Mesh

scene = Scene(
    cuboid=[
        Cuboid(name="table", dims=[1.0, 1.0, 0.1], pose=[0, 0, 0.5, 1, 0, 0, 0]),
    ],
    sphere=[
        Sphere(name="ball", radius=0.1, pose=[0.5, 0.5, 1.0, 1, 0, 0, 0])
    ],
    mesh=[
        Mesh(name="object", file_path="model.obj", pose=[0, 0, 0.6, 1, 0, 0, 0])
    ]
)
```

### Loading from YAML

Scene YAML files live in `curobo/content/configs/scene/`:

```yaml
cuboid:
  table:
    dims: [1.4, 1.4, 0.05]
    pose: [0.0, 0.0, -0.05, 1, 0, 0, 0.0]
  box:
    dims: [0.1, 0.1, 0.1]
    pose: [0.4, 0.0, 0.3, 1, 0, 0, 0.0]
```

Use via `MotionPlannerCfg.create(scene_model="collision_test.yml")` or load directly:

```python
from curobo.scene import Scene
scene = Scene.create("collision_test.yml")  # Resolves via ContentPath
```

### Runtime Scene Updates

Scenes can be updated without recreating the solver:

```python
# Move an existing obstacle
planner.scene_collision_checker.update_obstacle_pose("box", new_pose)

# Add a new obstacle
new_obstacle = Cuboid(name="new_box", dims=[0.1, 0.1, 0.1], pose=[...])
ik_solver.update_world(Scene(cuboid=[new_obstacle]))
```

The collision checker uses a cache system (`collision_cache={"cuboid": 10, "mesh": 5}`) to pre-allocate GPU buffers for dynamic obstacle additions. Without sufficient cache, adding obstacles beyond the cache size requires recreating the checker.

### Multi-Environment Scenes

For batched planning with different scenes per problem, set `multi_env=True` and `max_batch_size=N`:

```python
config = MotionPlannerCfg.create(
    robot="franka.yml",
    scene_model="collision_test.yml",
    max_batch_size=16,
    multi_env=True,
)
```

Each batched problem gets its own collision environment indexed by `idxs_env`.

## Constructing Optimization Problems

### Cost Architecture

Costs are implemented as Warp kernels (`curobo/_src/cost/`) with PyTorch tensor interfaces:

| Cost Class | Config | Purpose |
|-----------|--------|---------|
| `ToolPoseCost` | `ToolPoseCostCfg` | End-effector pose tracking (position + orientation) |
| `SceneCollisionCost` | `SceneCollisionCostCfg` | Robot-scene collision avoidance |
| `SelfCollisionCost` | `SelfCollisionCostCfg` | Robot self-collision avoidance |
| `CSpaceCost` | `CSpaceCostCfg` | Joint limits, velocity, acceleration, jerk penalties |
| `CSpaceDistCost` | `CSpaceDistCostCfg` | Distance from start/goal configuration |

These are aggregated by `RobotCostManager` (`curobo/_src/rollout/cost_manager/cost_manager_robot.py`) which the rollout calls during forward evaluation.

### Rollout Configuration

A `RobotRolloutCfg` defines the optimization problem by composing:

```python
RobotRolloutCfg(
    device_cfg=DeviceCfg(),
    transition_model_cfg=RobotStateTransitionCfg(...),  # How actions -> states
    cost_cfg=RobotCostManagerCfg(...),                    # Cost terms
    constraint_cfg=RobotCostManagerCfg(...),              # Hard constraints
    convergence_cfg=RobotCostManagerCfg(...),             # Convergence criteria
    scene_collision_cfg=SceneCollisionCfg(...),           # World obstacles
)
```

The rollout is created automatically by solver configs. For custom optimization, build it explicitly (see `curobo/examples/guides/custom_optimization.py`).

### Optimizers

Available in `curobo.optim`:

| Optimizer | Type | Best For |
|-----------|------|----------|
| `MPPI` | Particle (sampling) | Trajectory optimization, exploration |
| `EvolutionStrategies` | Particle (sampling) | Trajectory optimization |
| `LBFGSOpt` | Gradient | IK, fine trajectory refinement |
| `TorchOpt` | Gradient (PyTorch) | Custom differentiable costs |
| `ScipyOpt` | Gradient (SciPy) | CPU fallback |

Solvers typically chain optimizers via `MultiStageOptimizer`:
- **IK**: Levenberg-Marquardt seed solver → LBFGS refinement
- **TrajOpt**: Particle-based seed (MPPI/ES) → LBFGS refinement

### Custom Optimization Pipeline

For user-defined objectives, the pattern is:

1. Define a `Rollout` subclass (or use `RobotRollout`)
2. Configure an optimizer (`MPPI`, `LBFGSOpt`, etc.)
3. Wrap in `MultiStageOptimizer`
4. Call `optimizer.optimize(seed_action)`

See `curobo/examples/guides/custom_optimization.py` for a complete example using `RosenbrockRollout`.

### Pose Goal Specification

`GoalToolPose` supports **goalsets** (multiple candidate poses) and **multiple tool frames**:

```python
# Single goal for one tool frame
goal = GoalToolPose.from_poses(
    {"panda_hand": Pose(position=..., quaternion=...)},
    num_goalset=1,
)

# Goalset: 3 candidate poses for the same tool frame
positions = torch.zeros(1, 1, 1, 3, 3)  # (batch, horizon, env, goalset, xyz)
quaternions = torch.zeros(1, 1, 1, 3, 4)  # (batch, horizon, env, goalset, wxyz)
goalset = GoalToolPose(
    tool_frames=["panda_hand"],
    position=positions,
    quaternion=quaternions,
)
```

The solver automatically selects the best reachable pose from the goalset.

## Dual-Arm / Multi-Arm Planning

Multi-arm robots are modeled as a **single kinematic tree** with multiple end-effector frames. There is no special "multi-arm" API -- the same solvers work when configured with multiple tool frames.

### Robot Configuration

The `dual_ur10e.yml` config demonstrates the pattern:

```yaml
robot_cfg:
  kinematics:
    base_link: "world_base_link"
    tool_frames: ["tool1", "tool0"]  # Multiple end-effectors
    collision_link_names: [
      'shoulder_link', ..., 'tool0',
      'shoulder_link_1', ..., 'tool1'
    ]
    self_collision_ignore: {
      # Intra-arm ignores (same as single-arm)
      "upper_arm_link": ["shoulder_link", "forearm_link"],
      # Inter-arm: by default, all links between different arms collide
      # Add entries here to ignore specific cross-arm pairs
    }
    cspace:
      joint_names: [
        'shoulder_pan_joint', ..., 'wrist_3_joint',      # Arm 1
        'shoulder_pan_joint_1', ..., 'wrist_3_joint_1',  # Arm 2
      ]
```

Key points:
- **All joints** from all arms are in a single `joint_names` list
- **All collision links** are in `collision_link_names`
- `self_collision_ignore` handles both intra-arm adjacencies and inter-arm exceptions
- `tool_frames` defines which links are treated as end-effectors for IK/planning

### Specifying Goals for Multiple Arms

```python
# Different goals for each arm
goal_dict = {
    "tool0": Pose(position=arm1_pos, quaternion=arm1_quat),
    "tool1": Pose(position=arm2_pos, quaternion=arm2_quat),
}
goal = GoalToolPose.from_poses(goal_dict, num_goalset=1)

result = ik_solver.solve_pose(goal, current_state)
```

`GoalToolPose` stores poses per tool frame. When solving, the cost function computes pose error for **all specified tool frames simultaneously**.

### Reachability Analysis for Multi-Arm

The `--reachability` example supports multi-arm robots:

```bash
python -m curobo.examples.getting_started.inverse_kinematics --reachability --robot dual_ur10e.yml
```

This solves a dense grid of IK queries and visualizes the reachable workspace for each arm. The grid is evaluated for the **primary tool frame** (`tool_frames[0]`), with other tool frames held at their current pose.

### Grasp Planning with Multi-Arm

Grasp planning (`plan_grasp`) supports goalsets of candidate poses. For dual-arm manipulation, specify goalsets for each arm independently or jointly:

```python
# Each arm has its own set of candidate grasp poses
grasp_poses = GoalToolPose(
    tool_frames=["tool0", "tool1"],
    position=...,      # Shape: (..., n_grasps, 3)
    quaternion=...,    # Shape: (..., n_grasps, 4)
)

result = planner.plan_grasp(
    current_state=q_start,
    grasp_poses=grasp_poses,
    grasp_approach_offset=0.1,
    grasp_lift_offset=0.1,
)
```

## Robot Configuration Files

Robot configs are YAML files in `curobo/content/configs/robot/` with this structure:

```yaml
robot_cfg:
  kinematics:
    urdf_path: "robot/franka_description/franka_panda.urdf"
    asset_root_path: "robot/franka_description"  # Relative to content dir
    base_link: "panda_link0"
    tool_frames: ["panda_hand"]
    collision_link_names: [...]
    collision_spheres:
      link_name:
        - center: [x, y, z]
          radius: r
    self_collision_ignore: {link1: [link2, link3], ...}
    self_collision_buffer: {link1: 0.02, ...}
    lock_joints: {joint_name: fixed_position}  # e.g., gripper joints
    cspace:
      joint_names: [...]
      default_joint_position: [...]
      null_space_weight: [...]
      cspace_distance_weight: [...]
      max_jerk: 500.0
      max_acceleration: 15.0
  load_dynamics: false  # Set true to load inertial properties for dynamics
```

Collision spheres approximate each link's geometry. They are **required** for collision checking but not for pure kinematics. The `RobotBuilder` class (`curobo.robot_builder`) can auto-fit spheres from URDF meshes.

## Content Path Resolution

Config files are resolved via `curobo.content` which uses `importlib_resources` to locate files within the package:

```python
from curobo.content import get_robot_configs_path, get_scene_configs_path
# Returns pathlib.Path to the configs directories
```

User configs can be referenced by absolute path or placed in the package's content directory.

## Contributing Requirements

All commits must include a DCO sign-off (`git commit -s`). See `CONTRIBUTING.md` for the full DCO text.

Key test configuration from `pyproject.toml`:
- Tests run with `pytest-xdist` (`-n 4` by default)
- `--dist loadscope` for parallel test distribution
- CUDA device fixture: `cuda_device_cfg` (scope=session)
- `runtime.torch_compile = False` and `runtime.cuda_graph_reset = False` in test conftest
