# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dual-arm motion planning with relative pose constraint.

Workflow:
  1. Specify start poses and goal poses for both arms.
  2. Verify that both start and goal poses satisfy the relative-pose constraint.
  3. Solve IK (without relative-pose constraint) to find the initial joint state.
  4. Call the motion planner (with relative-pose constraint) to plan a trajectory.
"""

from __future__ import annotations

# Standard Library
import threading

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.solver.solver_core_cfg import SolverCoreCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState
from curobo._src.util.logging import setup_curobo_logger
from curobo.viewer import ViserVisualizer

wp.init()
setup_curobo_logger("error")

# ---------------------------------------------------------------------------
# Helper: relative-pose constraint verification
# ---------------------------------------------------------------------------

def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate [w, x, y, z] -> [w, -x, -y, -z]."""
    conj = q.clone()
    conj[..., 1:4] *= -1
    return conj


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions (wxyz)."""
    q1_w, q1_xyz = q1[..., 0:1], q1[..., 1:4]
    q2_w, q2_xyz = q2[..., 0:1], q2[..., 1:4]
    w = q1_w * q2_w - torch.sum(q1_xyz * q2_xyz, dim=-1, keepdim=True)
    xyz = q1_w * q2_xyz + q2_w * q1_xyz + torch.cross(q1_xyz, q2_xyz, dim=-1)
    return torch.cat([w, xyz], dim=-1)


def _rotate_vector(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate a 3D vector by a unit quaternion."""
    v_quat = torch.cat([torch.zeros_like(v[..., 0:1]), v], dim=-1)
    q_norm = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    return _quat_mul(_quat_mul(q_norm, v_quat), _quat_conj(q_norm))[..., 1:4]


def _axis_angle_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Axis-angle magnitude (rad) from a quaternion [w, x, y, z]."""
    w = q[..., 0]
    xyz = q[..., 1:4]
    norm = torch.norm(xyz, dim=-1)
    # Handle both short and long paths on the quaternion double cover
    angle = 2.0 * torch.atan2(norm, torch.abs(w))
    return angle


def _print_plan_failure_details(result) -> None:
    """Print detailed diagnostics for a failed planning result."""
    print("  --- Detailed failure diagnostics ---")
    print(f"  total_time: {result.total_time:.4f}s")
    print(f"  solve_time: {result.solve_time:.4f}s")

    if result.success is not None:
        print(f"  success (per seed): {result.success}")
    if result.feasible is not None:
        print(f"  feasible (per seed): {result.feasible}")
    if result.position_error is not None:
        print(f"  position_error (per seed): {result.position_error}")
    if result.rotation_error is not None:
        print(f"  rotation_error (per seed): {result.rotation_error}")
    if result.seed_cost is not None:
        print(f"  seed_cost: {result.seed_cost}")
    if result.cspace_error is not None:
        print(f"  cspace_error: {result.cspace_error}")

    # Check metrics for constraint violations
    if result.metrics is not None and result.metrics.costs_and_constraints is not None:
        cc = result.metrics.costs_and_constraints
        print("  --- Raw metrics costs ---")
        if hasattr(cc, "cost_names") and cc.cost_names:
            for name, val in zip(cc.cost_names, cc.cost_values):
                try:
                    print(f"    {name}: max={val.max().item():.6f}, mean={val.mean().item():.6f}")
                except Exception:
                    print(f"    {name}: {val}")
        print("  --- Constraint violations ---")
        if hasattr(cc, "constraint_names") and cc.constraint_names:
            for name, val in zip(cc.constraint_names, cc.constraint_values):
                try:
                    max_violation = val.max().item()
                    print(f"    {name}: max violation = {max_violation:.6f}")
                except Exception:
                    print(f"    {name}: {val}")
        if hasattr(cc, "get_feasible"):
            try:
                feasible_flat = cc.get_feasible(include_all_hybrid=False, sum_horizon=True)
                print(f"    overall feasible (flat): {feasible_flat}")
            except Exception:
                pass

    # Check interpolated metrics (this is what actually determines success!)
    if result.interpolated_metrics is not None and result.interpolated_metrics.costs_and_constraints is not None:
        cc = result.interpolated_metrics.costs_and_constraints
        print("  --- Interpolated metrics (determines success) ---")
        if hasattr(cc, "constraint_names") and cc.constraint_names:
            for name, val in zip(cc.constraint_names, cc.constraint_values):
                try:
                    max_violation = val.max().item()
                    print(f"    {name}: max violation = {max_violation:.6f}")
                except Exception:
                    print(f"    {name}: {val}")
        if hasattr(cc, "get_feasible"):
            try:
                feasible_flat = cc.get_feasible(include_all_hybrid=False, sum_horizon=True)
                print(f"    interpolated feasible: {feasible_flat}")
            except Exception:
                pass

    # Print trajectory (even if failed)
    if result.js_solution is not None:
        print("  --- Planned trajectory (last timestep joints) ---")
        last_pos = result.js_solution.position[0, 0, -1, :]
        print(f"    last_pos: {last_pos}")
        print(f"    horizon: {result.js_solution.position.shape[2]}")
        print(f"    dt: {result.js_solution.dt}")

    # Debug info (e.g., why IK or trajopt failed)
    if result.debug_info:
        print("  --- debug_info ---")
        for k, v in result.debug_info.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v}")
            else:
                print(f"    {k}: {v}")


def verify_relative_pose(
    primary_pose: Pose,
    secondary_pose: Pose,
    target_rel_pose: Pose,
    position_tol: float = 0.01,
    rotation_tol: float = 0.05,
) -> tuple[bool, float, float]:
    """Check whether two poses satisfy a relative-pose constraint.

    Returns:
        (is_satisfied, position_error_m, rotation_error_rad)
    """
    p_pri = primary_pose.position  # [B, 3]
    q_pri = primary_pose.quaternion  # [B, 4]
    p_sec = secondary_pose.position  # [B, 3]
    q_sec = secondary_pose.quaternion  # [B, 4]

    # Relative position in primary frame: R_pri^T * (p_sec - p_pri)
    rel_pos = _rotate_vector(p_sec - p_pri, _quat_conj(q_pri))

    # Relative quaternion: q_pri^* * q_sec
    rel_quat = _quat_mul(_quat_conj(q_pri), q_sec)

    # Position error
    target_rel_pos = target_rel_pose.position
    pos_err = torch.norm(rel_pos - target_rel_pos, dim=-1)

    # Rotation error: axis-angle of rel_quat * target_quat^{-1}
    target_rel_quat = target_rel_pose.quaternion
    target_quat_inv = _quat_conj(target_rel_quat)
    rot_diff = _quat_mul(rel_quat, target_quat_inv)
    rot_err = _axis_angle_from_quat(rot_diff)

    pos_err_val = pos_err.max().item()
    rot_err_val = rot_err.max().item()
    satisfied = pos_err_val <= position_tol and rot_err_val <= rotation_tol
    return satisfied, pos_err_val, rot_err_val


# ---------------------------------------------------------------------------
# Helper: inject relative-pose cost into solver core configs
# ---------------------------------------------------------------------------

def add_relative_pose_to_solver_core(
    core_cfg: SolverCoreCfg,
    relative_pose_cfg: RelativePoseCostCfg,
) -> None:
    """Inject relative_pose_cfg into all cost managers of a SolverCoreCfg."""
    for rollout_cfg in core_cfg.optimizer_rollout_configs:
        _add_to_rollout(rollout_cfg, relative_pose_cfg)
    if core_cfg.metrics_rollout_config is not None:
        _add_to_rollout(core_cfg.metrics_rollout_config, relative_pose_cfg)


def _add_to_rollout(
    rollout_cfg: RobotRolloutCfg,
    relative_pose_cfg: RelativePoseCostCfg,
) -> None:
    for manager_cfg in rollout_cfg.get_cost_manager_configs(include_constraint_cfg=True):
        manager_cfg.relative_pose_cfg = relative_pose_cfg


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------

def example_with_motion_planner():
    device_cfg = DeviceCfg()

    target_rel_pose = Pose(
        position=torch.tensor([0.3, -0.2, 0.1], device=device_cfg.device),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device_cfg.device),
    )  # Desired pose of panda_right_hand relative to panda_left_hand in panda_left_hand's frame

    start_poses = {
        "panda_left_hand": Pose(
            position=torch.tensor([[0.2, 0.0, 0.5]], device=device_cfg.device),
            quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device_cfg.device),
        ),
        "panda_right_hand": Pose(
            position=torch.tensor([[0.5, 0.2, 0.4]], device=device_cfg.device),
            quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device_cfg.device),
        ),
    }
    goal_poses = {
        "panda_left_hand": Pose(
            position=torch.tensor([[0.3, 0.0, 0.6]], device=device_cfg.device),
            quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device_cfg.device),
        ),
        "panda_right_hand": Pose(
            position=torch.tensor([[0.6, 0.2, 0.5]], device=device_cfg.device),
            quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device_cfg.device),
        ),
    }
    # =====================================================================
    # Step 0: Build the relative-pose constraint specification
    # =====================================================================
    # dual_franka.yml tool_frames: ["panda_left_hand", "panda_right_hand"]
    relative_pose_cfg = RelativePoseCostCfg(
        device_cfg=device_cfg,
        weight=torch.tensor([50000.0, 5000.0]),
        primary_tool_frame="panda_left_hand",
        secondary_tool_frame="panda_right_hand",
        target_relative_pose=target_rel_pose,
        position_tolerance=0.01,
        orientation_tolerance=0.05,
    )

    # =====================================================================
    # Step 1: Verify start poses
    # =====================================================================
    print("=== Step 1: Verifying start poses satisfy relative-pose constraint ===")
    ok, p_err, r_err = verify_relative_pose(
        start_poses["panda_left_hand"],
        start_poses["panda_right_hand"],
        target_rel_pose,
        position_tol=0.01,
        rotation_tol=0.05,
    )
    print(f"  Start pose relative error: pos={p_err:.4f} m, rot={r_err:.4f} rad")
    if not ok:
        print("  ERROR: start poses violate the relative-pose constraint!")
        return
    print("  Start poses OK.")

    # =====================================================================
    # Step 2: Solve IK (no relative-pose constraint) to find initial state
    # =====================================================================
    print("\n=== Step 2: Solving IK for initial joint state ===")
    ik_cfg = InverseKinematicsCfg.create(
        robot="dual_franka.yml",
        num_seeds=32,
    )
    ik_solver = InverseKinematics(ik_cfg)

    start_goal = GoalToolPose.from_poses(start_poses, num_goalset=1)
    ik_result = ik_solver.solve_pose(start_goal)

    if ik_result is None or not ik_result.success.any():
        print("  ERROR: IK failed to find an initial configuration!")
        return

    start_state = ik_result.js_solution.clone().squeeze(1)
    print(f"  IK succeeded. Initial joint state shape: {start_state.position.shape}")
    print(f"  IK position error: {ik_result.position_error.item() * 1000:.3f} mm")

    ik_solver.destroy()

    # =====================================================================
    # Step 3: Verify goal poses for both arms
    # =====================================================================

    print("\n=== Step 3: Verifying goal poses satisfy relative-pose constraint ===")
    ok, p_err, r_err = verify_relative_pose(
        goal_poses["panda_left_hand"],
        goal_poses["panda_right_hand"],
        target_rel_pose,
        position_tol=0.01,
        rotation_tol=0.05,
    )
    print(f"  Goal pose relative error: pos={p_err:.4f} m, rot={r_err:.4f} rad")
    if not ok:
        print("  ERROR: goal poses violate the relative-pose constraint!")
        return
    print("  Goal poses OK.")

    # =====================================================================
    # Step 4: Build motion planner with relative-pose constraint
    # =====================================================================
    print("\n=== Step 4: Building motion planner with relative-pose constraint ===")
    planner_cfg = MotionPlannerCfg.create(
        robot="dual_franka.yml",
        scene_model="collision_test.yml",
        num_ik_seeds=32,
        num_trajopt_seeds=4,
    )

    # Inject relative-pose cost into the planner's TrajOpt solver cores
    add_relative_pose_to_solver_core(
        planner_cfg.trajopt_solver_config.core_cfg,
        relative_pose_cfg,
    )

    planner = MotionPlanner(planner_cfg)
    print("Warming up planner...")
    planner.warmup(enable_graph=False, num_warmup_iterations=1)
    print("Warmup done.")

    # =====================================================================
    # Step 4.5: Viser visualization (before planning)
    # =====================================================================
    print("\n=== Step 4.5: Launching Viser visualization ===")
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file="dual_franka.yml"),
        connect_ip="0.0.0.0",
        connect_port=8080,
        add_robot_to_scene=True,
        add_control_frames=False,
        visualize_robot_spheres=False,
    )

    # Add collision scene from planner config
    scene_cfg = planner_cfg.scene_collision_cfg.scene_model
    viser_viz.add_scene(scene_cfg)

    # Visualize IK-derived start configuration
    viser_viz.set_joint_state(start_state)

    # Add start pose frames (green)
    for frame_name, pose in start_poses.items():
        viser_viz.add_frame(f"/start_{frame_name}", pose, scale=0.05)

    # Add goal pose frames (red)
    for frame_name, pose in goal_poses.items():
        viser_viz.add_frame(f"/goal_{frame_name}", pose, scale=0.05)

    print("  Viser running at http://localhost:8080")
    print("  Green frames = start poses, Red frames = goal poses")

    # Add a "Move" button and wait for user click before planning
    move_event = threading.Event()

    def on_move_clicked(_):
        move_event.set()
        print("  [Move] button clicked — starting planning...")

    move_btn = viser_viz._server.gui.add_button("Move", color="green")
    move_btn.on_click(on_move_clicked)

    print("  Waiting for user to click the 'Move' button in Viser...")
    move_event.wait()

    # =====================================================================
    # Step 5: Plan from IK-derived start state to goal poses
    # =====================================================================
    print("\n=== Step 5: Motion planning ===")
    goal = GoalToolPose.from_poses(goal_poses, num_goalset=1)
    active_js = planner.kinematics.get_active_js(start_state)

    result = planner.plan_pose(
        goal,
        active_js,
        max_attempts=3,
    )

    if result is not None and result.success.any():
        print(f"Planning succeeded in {result.total_time:.3f}s")
        print(f"Position error: {result.position_error.item():.4f} m")
        print(f"Rotation error: {result.rotation_error.item():.4f} rad")
    else:
        print("Planning failed")
        if result is not None:
            _print_plan_failure_details(result)
        else:
            print("No result returned from planner.")

    planner.destroy()


if __name__ == "__main__":
    example_with_motion_planner()
