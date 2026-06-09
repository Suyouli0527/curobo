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
| `RelativePoseCost` | `RelativePoseCostCfg` | Fixed relative pose between two end-effectors (dual-arm) |

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

## Motion Planner Internals: `plan_pose()` Execution Flow

`MotionPlanner.plan_pose()` (`curobo/_src/motion/motion_planner.py:207`) implements a **two-stage pipeline**: IK finds a collision-free goal configuration, then TrajOpt smooths a trajectory from the current state to that goal. Both stages are GPU-parallel and batched.

### High-Level Pipeline

```
plan_pose(goal, current_state)
  └── _plan_pose_single()                    [motion_planner.py:233]
        ├── Stage 1: ik_solver.solve_pose()   [solver_ik.py:631]
        │     └── returns: seed_config [batch, num_seeds, dof]
        └── Stage 2: trajopt_solver.solve_pose() [solver_trajopt.py:680]
              └── seeded with ik_result.solution
```

Both solvers retry up to `max_attempts` times. TrajOpt can also use graph-planner seeds (`enable_graph_attempt`) for difficult problems.

### Stage 1: IK — Finding a Collision-Free Goal Configuration

**Entry:** `IKSolver.solve_pose()` → `_solve_impl()` (`solver_ik.py:363`)

1. **Optional LM seed generation** (`use_lm_seed=True`): a fast Levenberg-Marquardt solver (`SeedIKSolver`) generates high-quality initial guesses before the main optimizer.
2. **Prepare action seeds**: `SeedManager.prepare_action_seeds()` reshapes seeds to `(batch*num_seeds, 1, dof)`.
3. **Run optimizer**: `MultiStageOptimizer.optimize()` chains:
   - **Stage 1 (MPPI/ES)**: particle-based global exploration
   - **Stage 2 (LBFGS)**: gradient-based refinement of the best particle
4. **Compute metrics**: `metrics_rollout.compute_metrics_from_action()` evaluates pose error, collision, and convergence.
5. **Build result**: `_get_result()` selects top-k seeds by cost.

### Stage 2: TrajOpt — Time-Optimal Trajectory Refinement

**Entry:** `TrajOptSolver.solve_pose()` → `_solve_impl()` (`solver_trajopt.py:258`)

1. **Prepare trajectory seeds**: `SeedManager.prepare_trajectory_seeds()` interpolates from `current_state` to the IK seed configs across the trajectory horizon.
2. **Time-optimal finetune loop**: iteratively adjusts `dt` (trajectory segment duration):
   - Iteration 0: coarse optimization with full optimizer chain
   - Iteration 1+: LBFGS-only refinement with scaled `dt`
   - Best solution is kept across iterations (by `dt` and success)
3. **Interpolate**: the optimized B-spline is resampled to a dense trajectory for execution.

### Action → State → Cost Data Flow

Both IK and TrajOpt use the same `RobotRollout` pipeline:

```
act_seq [B, H, D]
  └── RobotStateTransition.forward()
        ├── state integration (CUDA kernel)
        │     POSITION    → StateFromPositionClique
        │     ACCELERATION → StateFromAcceleration
        │     BSPLINE     → StateFromBSplineKnot
        └── compute_augmented_state()
              ├── Forward kinematics → tool poses, collision spheres
              └── Inverse dynamics   → joint torques (optional)
  └── RobotCostManager.compute_costs()
        ├── ToolPoseCost      (parallel CUDA stream)
        ├── SceneCollisionCost (parallel CUDA stream)
        ├── SelfCollisionCost  (parallel CUDA stream)
        ├── CSpaceCost         (parallel CUDA stream)
        └── ... (synchronize streams)
  └── RolloutMetrics
        ├── costs_and_constraints: CostCollection + CostCollection
        ├── feasible: bool [B]  (all constraints <= 0)
        └── convergence: CostCollection (pose tolerances)
```

**Key design**: optimization and metrics use *separate* transition models and cost managers. This prevents CUDA graph invalidation when convergence checks (which need different tolerances) are run.

### Cost Classes — Mathematical Formulations

| Cost | Formula | Purpose |
|------|---------|---------|
| **ToolPoseCost** | `0.5 * w_pos * \|p - p_goal\|² + w_rot * \|axis_angle(q⁻¹ ⊗ q_goal)\|²` | End-effector pose tracking |
| **SceneCollisionCost** | `w * Σ max(0, activation_dist - sphere_dist)` | Robot-scene collision avoidance |
| **SelfCollisionCost** | `w * Σ max(0, - (dist - r_i - r_j - padding))` | Robot self-collision avoidance |
| **CSpaceCost (POSITION)** | `0.5 * w * \|q - q_target\|² + bound_penalty` | Joint limits + position regularization |
| **CSpaceCost (STATE)** | `Σ 0.5 * w_i * \|state_i\|²` for vel/acc/jerk/torque/energy | Full state smoothness |
| **CSpaceDistCost** | `w * \|q - q_goal\|²` | Distance to start/goal config |
| **RelativePoseCost** | `0.5 * w_pos * \|rel_pos - target\|² + w_rot * \|axis_angle(rel_quat ⊗ target⁻¹)\|²` | Dual-arm relative pose |

With `convert_to_binary=True` (constraints), costs become classification:
- `0` = feasible (no violation)
- `> 0` = violated (magnitude = severity)

### Cost vs Constraint vs Convergence

The rollout distinguishes three categories:

| Category | Config | Behavior | Examples |
|----------|--------|----------|----------|
| **Cost** | `cost_cfg` | Smooth penalty minimized by optimizer | Pose error, c-space distance, smoothness |
| **Constraint** | `constraint_cfg` | Binary feasibility (`convert_to_binary=True`) | Collision, joint limits |
| **Convergence** | `convergence_cfg` | Tolerance check for termination | Position tolerance, orientation tolerance |

Success = `feasible AND converged`. Feasibility means all constraints are satisfied (values ≤ 0). Convergence means all tolerance metrics are below their thresholds.

### Default Weights from YAML Configs

#### Metrics (evaluation only)

```yaml
# metrics_base.yml
constraint_cfg:
  scene_collision_cfg: { weight: 5000.0, convert_to_binary: True }
  self_collision_cfg: { weight: 5000.0, convert_to_binary: True }
  cspace_cfg: { weight: [5000, 5000, 5000, 5000, 5000], cost_type: STATE }
convergence_cfg:
  tool_pose_cfg: { weight: [1.0, 1.0] }
  target_cspace_dist_cfg: { weight: 0.0 }
  start_cspace_dist_cfg: { weight: 0.001 }
```

#### IK

```yaml
# ik/lbfgs_ik.yml
cost_cfg:
  cspace_cfg: { weight: [5000.0, 1000.0], cost_type: POSITION }
  tool_pose_cfg: { weight: [10000.0, 500.0] }
constraint_cfg:
  scene_collision_cfg: { weight: 5000.0 }
  self_collision_cfg: { weight: 5000.0 }

# ik/particle_ik.yml
cost_cfg:
  cspace_cfg: { weight: [50.0, 1.0], cost_type: POSITION }
  tool_pose_cfg: { weight: [1000.0, 10.0] }
constraint_cfg:
  scene_collision_cfg: { weight: 500.0 }
  self_collision_cfg: { weight: 50.0 }
```

#### Trajectory Optimization

```yaml
# trajopt/lbfgs_bspline_trajopt.yml
constraint_cfg:
  scene_collision_cfg: { weight: 100000.0, use_sweep: true, use_speed_metric: true }
  self_collision_cfg: { weight: 10000.0 }
cost_cfg:
  cspace_cfg:
    weight: [10000.0, 10000.0, 100.0, 50.0, 100.0]
    cost_type: STATE
    squared_l2_regularization_weight: [1000.0, 10000.0, 5.0, 0.0, 10000.0]
  tool_pose_cfg:
    weight: [1000000.0, 100000.0]
    _non_terminal_pose_axes_weight_factor: [0, 0, 0, 0, 0, 0]

# trajopt/particle_trajopt.yml
constraint_cfg:
  scene_collision_cfg: { weight: 10000.0 }
  self_collision_cfg: { weight: 1000.0 }
cost_cfg:
  cspace_cfg: { weight: [1.0, 0, 0, 0, 0], cost_type: STATE }
  tool_pose_cfg: { weight: [0.0, 0.0] }
```

**Note**: Weights are significantly higher in TrajOpt than IK because TrajOpt must balance pose tracking against smoothness and collision over a full horizon, while IK only optimizes a single configuration.

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

### Relative Pose Constraint (Dual-Arm)

For tasks that require maintaining a fixed relative pose between two arms (e.g., jointly carrying an object), use `RelativePoseCost`:

```python
from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg

relative_pose_cfg = RelativePoseCostCfg(
    device_cfg=device_cfg,
    weight=torch.tensor([1.0, 0.5]),  # [position_weight, rotation_weight]
    primary_tool_frame="panda_hand_1",
    secondary_tool_frame="panda_hand",
    target_relative_pose=Pose(
        position=torch.tensor([0.3, 0.0, -0.1]),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    ),
)

# Inject into planner's IK and TrajOpt solver cores
for rollout_cfg in planner_cfg.ik_solver_config.core_cfg.optimizer_rollout_configs:
    rollout_cfg.cost_cfg.relative_pose_cfg = relative_pose_cfg
for rollout_cfg in planner_cfg.trajopt_solver_config.core_cfg.optimizer_rollout_configs:
    rollout_cfg.cost_cfg.relative_pose_cfg = relative_pose_cfg
```

The cost computes the relative pose of `secondary` w.r.t. `primary`'s frame and penalizes deviation from the target. `weight` follows the same `[pos, rot]` convention as `ToolPoseCost`.

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

### Dual-Arm Franka with Wrist Camera

A variant `dual_franka_with_camera.yml` / `dual_franka_with_camera.urdf` adds a `wrist_camera` link between the flange (`link8`) and the hand on each arm. The camera mounting plate is 3 mm thick, shifting the hand 3 mm in the local z direction relative to the camera.

**Kinematic chain per arm:**
```
link8 ──[rot=-45°]──► wrist_camera ──[z=+0.003m]──► hand
```

- `wrist_camera_joint` (fixed): connects `link8` → `wrist_camera`, same transform as the original `hand_joint` (`rpy="0 0 -0.785398163397"`, `xyz="0 0 0"`)
- `hand_joint` (fixed): connects `wrist_camera` → `hand`, with an additional 3 mm z offset (`xyz="0 0 0.003"`)

The camera link loads `meshes/visual/wrist_camera.STL` for visualization but is **not** included in `collision_link_names`.

**Key differences from `dual_franka.yml`:**
- `urdf_path` points to `dual_franka_with_camera.urdf`
- `mesh_link_names` includes `panda_right_wrist_camera` and `panda_left_wrist_camera` for rendering
- Hand collision spheres remain unchanged (they are defined in the `hand` link frame)

Use it the same way as any other robot config:

```python
config = MotionPlannerCfg.create(
    robot="dual_franka_with_camera.yml",
    scene_model="collision_test.yml",
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

## Coding Standards and LLM Development Guidelines

The following rules are derived from the project's LLM prompt rule files (`design_principles.mdc`, `package_format_rules.mdc`, `file_naming.mdc`, `documenter.mdc`). They ensure generated code matches the structure, naming, and documentation standards of human-written code.

### Design Principles

Follow these principles when adding or modifying code:

- **DRY (Don't Repeat Yourself)**: Avoid code duplication; use reusable components.
- **KISS (Keep It Simple, Stupid)**: Keep designs simple and avoid unnecessary complexity.
- **YAGNI (You Aren't Gonna Need It)**: Implement only necessary features.
- **Separation of Concerns**: Divide code into distinct sections with single responsibilities.
- **Composition Over Inheritance**: Prefer composition for code reuse.
- **Encapsulation**: Hide internal state and expose only necessary interfaces.
- **Single Responsibility Principle (SRP)**: One class, one responsibility.
- **Open/Closed Principle (OCP)**: Open for extension, closed for modification.
- **Interface Segregation Principle (ISP)**: Specific interfaces for different clients.
- **Dependency Inversion Principle (DIP)**: Depend on abstractions, not concretions.
- **Law of Demeter**: Interact only with immediate dependencies.
- **Optimize for Performance**: Use profiling tools to identify bottlenecks.
- **Use Virtual Environments**: Isolate project dependencies.

### File Naming Convention (Category-First)

All new files must follow the **category-first** naming convention:

```
category_specific.py
```

- **category**: Identifies the broader component type or subsystem.
- **specific**: Describes the particular implementation or variant.

Examples:
- `connector_linear.py` (NOT `linear_connector.py`)
- `sampler_node.py` (NOT `node_sampler.py`)
- `connector_cubic_spline.py` (NOT `connector_spline_cubic.py`) -- multiple adjectives go from general to specific.

Directory structure and imports should reflect this organization:
- File: `/connectors/connector_linear.py`
- Import: `from curobo.connectors.connector_linear import LinearConnector`
- Class name: `LinearConnector` (CamelCase, specific-type first)

### Python Code Style

- Follow **flake8** rules as configured in `setup.cfg`.
- Use **descriptive names** for functions, classes, and other members.
- Use **type hints** for all function signatures.
- Follow **PEP 563** (Postponed Evaluation of Annotations): use `from __future__ import annotations`.
- Use the project's **logging utilities** for prints and warnings (do not use bare `print()` for diagnostics).
- Avoid using deprecated functions.

### Docstrings

- Always docstring in the same line as `'''` or `"""`.
- Use `:func:`, `:class:`, `:attr:` for references.
- Write shapes of inputs and outputs when possible (e.g., `(batch, horizon, dof)`).
- For dataclasses, document attributes **as they are declared** (not in the class docstring).
- Follow the **Google docstring convention**.

### Writing Tests

- Tests live under `src/curobo/tests/` (or equivalent test folder).
- Do **not** write unit tests for `wp.kernel` decorators (Warp GPU kernels).
- When writing gradient checking with `gradcheck`, use **float32** (not float64).

### Documentation Principles

- Write **concise and clear** documentation; tailor it to the knowledge level of new users.
- Provide **code snippets and examples** for APIs.
- Use **cross-references** to link related sections and API docs.
- Follow **reST and Sphinx conventions**.
- Maintain **consistency** in terminology, formatting, and style.
- Line width for docs: **99 characters**.

## Contributing Requirements

All commits must include a DCO sign-off (`git commit -s`). See `CONTRIBUTING.md` for the full DCO text.

Key test configuration from `pyproject.toml`:
- Tests run with `pytest-xdist` (`-n 4` by default)
- `--dist loadscope` for parallel test distribution
- CUDA device fixture: `cuda_device_cfg` (scope=session)
- `runtime.torch_compile = False` and `runtime.cuda_graph_reset = False` in test conftest
