# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive dual-arm coordinated motion planning.

Run:
    python -m dual_panda_benchmark.dual_arm_relative_pose

Open Viser, drag the left/right target frames for ordinary dual-arm planning, or click
``Coordinate`` to record the current relative pose and drive both arms with one cooperative frame.
"""

from __future__ import annotations

# Standard Library
import argparse
import threading
import time
from pathlib import Path

# Third Party
import torch
import warp as wp
import yaml

# CuRobo
from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.solver.solver_core_cfg import SolverCoreCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import setup_curobo_logger
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import ContentPath, GoalToolPose, JointState
from curobo.viewer import ViserVisualizer

wp.init()
setup_curobo_logger("error")

LEFT_TOOL_FRAME = "panda_left_hand_tcp"
RIGHT_TOOL_FRAME = "panda_right_hand_tcp"
COORDINATE_FRAME = "/target_coordinate_frame"


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Return quaternion conjugate for ``[w, x, y, z]`` tensors."""
    conj = q.clone()
    conj[..., 1:4] *= -1
    return conj


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Return Hamilton product for ``[w, x, y, z]`` tensors."""
    q1_w, q1_xyz = q1[..., 0:1], q1[..., 1:4]
    q2_w, q2_xyz = q2[..., 0:1], q2[..., 1:4]
    w = q1_w * q2_w - torch.sum(q1_xyz * q2_xyz, dim=-1, keepdim=True)
    xyz = q1_w * q2_xyz + q2_w * q1_xyz + torch.cross(q1_xyz, q2_xyz, dim=-1)
    return torch.cat([w, xyz], dim=-1)


def _rotate_vector(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit quaternions."""
    v_quat = torch.cat([torch.zeros_like(v[..., 0:1]), v], dim=-1)
    q_norm = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    return _quat_mul(_quat_mul(q_norm, v_quat), _quat_conj(q_norm))[..., 1:4]


def _axis_angle_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Return axis-angle magnitude in radians from ``[w, x, y, z]`` quaternions."""
    w = q[..., 0]
    xyz = q[..., 1:4]
    norm = torch.norm(xyz, dim=-1)
    return 2.0 * torch.atan2(norm, torch.abs(w))


def _relative_pose(primary_pose: Pose, secondary_pose: Pose) -> Pose:
    """Return secondary pose expressed in the primary pose frame."""
    rel_position = _rotate_vector(
        secondary_pose.position - primary_pose.position,
        _quat_conj(primary_pose.quaternion),
    )
    rel_quaternion = _quat_mul(_quat_conj(primary_pose.quaternion), secondary_pose.quaternion)
    return Pose(position=rel_position, quaternion=rel_quaternion)


def _cooperative_pose_from_tool_poses(left_pose: Pose, right_pose: Pose) -> Pose:
    """Return the cooperative frame pose.

    Convention: cooperative orientation equals the left TCP orientation, and cooperative position
    is the midpoint between the left and right TCP positions.
    """
    return Pose(
        position=0.5 * (left_pose.position + right_pose.position),
        quaternion=left_pose.quaternion.clone(),
    )


def _tool_poses_from_cooperative_pose(cooperative_pose: Pose, target_rel_pose: Pose) -> dict[str, Pose]:
    """Convert a cooperative frame pose to left/right TCP goal poses."""
    half_rel_position_world = 0.5 * _rotate_vector(
        target_rel_pose.position,
        cooperative_pose.quaternion,
    )
    left_pose = Pose(
        position=cooperative_pose.position - half_rel_position_world,
        quaternion=cooperative_pose.quaternion.clone(),
    )
    right_pose = Pose(
        position=cooperative_pose.position + half_rel_position_world,
        quaternion=_quat_mul(cooperative_pose.quaternion, target_rel_pose.quaternion),
    )
    return {
        LEFT_TOOL_FRAME: left_pose,
        RIGHT_TOOL_FRAME: right_pose,
    }


def _pose_from_handle(handle, device_cfg: DeviceCfg) -> Pose:
    """Build a :class:`Pose` from a Viser transform-control handle."""
    return Pose.from_numpy(handle.position, handle.wxyz, device_cfg=device_cfg)


def _set_handle_pose(handle, pose: Pose) -> None:
    """Update a Viser transform-control handle from a :class:`Pose`."""
    handle.position = pose.position.detach().cpu().squeeze().numpy()
    handle.wxyz = pose.quaternion.detach().cpu().squeeze().numpy()


def _print_plan_failure_details(result) -> None:
    """Print detailed diagnostics for a failed planning result."""
    print("  --- Detailed failure diagnostics ---")
    print(f"  total_time: {result.total_time:.4f}s")
    print(f"  solve_time: {result.solve_time:.4f}s")
    if result.success is not None:
        print(f"  success: {result.success}")
    if result.feasible is not None:
        print(f"  feasible: {result.feasible}")
    if result.position_error is not None:
        print(f"  position_error: {result.position_error}")
    if result.rotation_error is not None:
        print(f"  rotation_error: {result.rotation_error}")
    if result.seed_cost is not None:
        print(f"  seed_cost: {result.seed_cost}")

    for metrics_name, metrics in [
        ("raw", result.metrics),
        ("interpolated", result.interpolated_metrics),
    ]:
        if metrics is None or metrics.costs_and_constraints is None:
            continue
        cc = metrics.costs_and_constraints
        print(f"  --- {metrics_name} costs ---")
        for name, value in zip(cc.costs.names, cc.costs.values):
            print(f"    {name}: max={value.max().item():.6f}, mean={value.mean().item():.6f}")
        print(f"  --- {metrics_name} constraints ---")
        for name, value in zip(cc.constraints.names, cc.constraints.values):
            print(f"    {name}: max={value.max().item():.6f}, mean={value.mean().item():.6f}")
        feasible = cc.get_feasible(include_all_hybrid=False, sum_horizon=True)
        print(f"    feasible: {feasible}")


def _trajectory_relative_pose_error(
    planner: MotionPlanner,
    trajectory: JointState,
    target_rel_pose: Pose,
) -> tuple[float, float, float, float]:
    """Return max/mean relative-pose errors over an interpolated trajectory."""
    traj = trajectory.squeeze(0)
    position = traj.position
    while position.ndim > 2:
        position = position[0]

    trajectory_state = JointState.from_position(position, joint_names=traj.joint_names)
    kin_state = planner.compute_kinematics(planner.kinematics.get_active_js(trajectory_state))
    rel_pose = _relative_pose(
        kin_state.tool_poses[LEFT_TOOL_FRAME],
        kin_state.tool_poses[RIGHT_TOOL_FRAME],
    )

    position_error = torch.norm(rel_pose.position - target_rel_pose.position, dim=-1)
    rotation_error = _axis_angle_from_quat(
        _quat_mul(rel_pose.quaternion, _quat_conj(target_rel_pose.quaternion))
    )
    return (
        position_error.max().item(),
        position_error.mean().item(),
        rotation_error.max().item(),
        rotation_error.mean().item(),
    )


def add_relative_pose_to_solver_core(
    core_cfg: SolverCoreCfg,
    relative_pose_cfg: RelativePoseCostCfg,
) -> None:
    """Register relative-pose tracking as both cost and trajopt feasibility constraint."""
    for rollout_cfg in core_cfg.optimizer_rollout_configs:
        _add_to_rollout_cost_and_constraint(rollout_cfg, relative_pose_cfg)
    if core_cfg.metrics_rollout_config is not None:
        _add_to_rollout_cost_and_constraint(core_cfg.metrics_rollout_config, relative_pose_cfg)


def _add_to_rollout_cost_and_constraint(
    rollout_cfg: RobotRolloutCfg,
    relative_pose_cfg: RelativePoseCostCfg,
) -> None:
    if rollout_cfg.cost_cfg is not None:
        rollout_cfg.cost_cfg.relative_pose_cfg = relative_pose_cfg
    if rollout_cfg.constraint_cfg is not None:
        rollout_cfg.constraint_cfg.relative_pose_cfg = relative_pose_cfg


def _iter_trajopt_rollouts(planner: MotionPlanner) -> list[RobotRollout]:
    """Return all trajectory-optimization rollouts that may contain relative-pose costs."""
    core = planner.trajopt_solver.core
    rollouts = [core.metrics_rollout, core.auxiliary_rollout]
    rollouts.extend(core.optimizer_rollouts)
    rollouts.extend(core.additional_metrics_rollouts.values())
    return rollouts


def _set_relative_pose_tracking(
    planner: MotionPlanner,
    enabled: bool,
    target_rel_pose: Pose | None = None,
) -> None:
    """Enable or disable relative-pose costs across all trajopt rollouts."""
    for rollout in _iter_trajopt_rollouts(planner):
        for manager in rollout._cost_manager_list:
            if manager.has_cost("relative_pose"):
                cost = manager.get_cost("relative_pose")
                if target_rel_pose is not None:
                    cost.target_relative_pose = target_rel_pose.clone()
                    cost.config.target_relative_pose = target_rel_pose.clone()
                if enabled:
                    cost.enable_cost()
                else:
                    cost.disable_cost()


def _get_current_tool_poses(planner: MotionPlanner, current_state: JointState) -> dict[str, Pose]:
    """Compute the current left/right TCP poses from the current joint state."""
    kin_state = planner.compute_kinematics(planner.kinematics.get_active_js(current_state.clone()))
    return {
        LEFT_TOOL_FRAME: kin_state.tool_poses[LEFT_TOOL_FRAME],
        RIGHT_TOOL_FRAME: kin_state.tool_poses[RIGHT_TOOL_FRAME],
    }


def _build_planner(args, device_cfg: DeviceCfg, relative_pose_cfg: RelativePoseCostCfg):
    env_path = Path(args.env) if args.env else Path(__file__).parent / "config" / "collision_env.yml"
    with open(env_path) as file:
        scene_dict = yaml.safe_load(file)

    planner_cfg = MotionPlannerCfg.create(
        robot=args.robot,
        scene_model=scene_dict,
        self_collision_check=True,
        num_ik_seeds=args.num_ik_seeds,
        num_trajopt_seeds=args.num_trajopt_seeds,
        collision_cache={"cuboid": 20},
    )
    add_relative_pose_to_solver_core(planner_cfg.trajopt_solver_config.core_cfg, relative_pose_cfg)
    _add_to_rollout_cost_and_constraint(
        planner_cfg.trajopt_solver_config.metrics_rollout_config,
        relative_pose_cfg,
    )
    planner = MotionPlanner(planner_cfg)
    _set_relative_pose_tracking(planner, enabled=False)
    return planner, Scene.create(scene_dict)


def main() -> None:
    """Run the interactive coordinated planning demo."""
    parser = argparse.ArgumentParser(description="Interactive dual-arm coordinated planning")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--robot", type=str, default="dual_franka.yml", help="Robot config file")
    parser.add_argument("--env", type=str, default=None, help="Collision environment YAML")
    parser.add_argument("--num-ik-seeds", type=int, default=32, help="Number of IK seeds")
    parser.add_argument("--num-trajopt-seeds", type=int, default=4, help="Number of trajopt seeds")
    parser.add_argument(
        "--relative-position-weight",
        type=float,
        default=50000.0,
        help="Relative-pose position cost weight used in coordinate mode.",
    )
    parser.add_argument(
        "--relative-rotation-weight",
        type=float,
        default=5000.0,
        help="Relative-pose rotation cost weight used in coordinate mode.",
    )
    parser.add_argument(
        "--relative-position-tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed relative-pose translation error in coordinate mode.",
    )
    parser.add_argument(
        "--relative-orientation-tolerance",
        type=float,
        default=0.05,
        help="Maximum allowed relative-pose rotation error in coordinate mode.",
    )
    parser.add_argument(
        "--reject-relative-violations",
        action="store_true",
        help="Reject coordinate-mode trajectories whose interpolated relative-pose error exceeds tolerance.",
    )
    args = parser.parse_args()

    device_cfg = DeviceCfg()
    relative_pose_cfg = RelativePoseCostCfg(
        device_cfg=device_cfg,
        weight=torch.tensor(
            [args.relative_position_weight, args.relative_rotation_weight],
            device=device_cfg.device,
        ),
        primary_tool_frame=LEFT_TOOL_FRAME,
        secondary_tool_frame=RIGHT_TOOL_FRAME,
        target_relative_pose=Pose(
            position=torch.zeros(3, device=device_cfg.device),
            quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device_cfg.device),
        ),
        position_tolerance=args.relative_position_tolerance,
        orientation_tolerance=args.relative_orientation_tolerance,
    )

    print(f"Initializing MotionPlanner with robot={args.robot}...")
    planner, scene = _build_planner(args, device_cfg, relative_pose_cfg)
    print("Warming up planner...")
    planner.warmup(enable_graph=False, num_warmup_iterations=5)

    print("Initializing Viser...")
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=args.robot),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_robot_to_scene=True,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )
    viser_viz.add_scene(scene, add_control_frames=False)
    server = viser_viz._server

    state = {
        "current_state": planner.default_joint_state.clone().unsqueeze(0),
        "is_moving": False,
        "coordinate_mode": False,
        "target_rel_pose": None,
    }
    viser_viz.set_joint_state(state["current_state"].squeeze(0))

    current_tool_poses = _get_current_tool_poses(planner, state["current_state"])
    cooperative_pose = _cooperative_pose_from_tool_poses(
        current_tool_poses[LEFT_TOOL_FRAME],
        current_tool_poses[RIGHT_TOOL_FRAME],
    )
    cooperative_frame = viser_viz.add_control_frame(
        COORDINATE_FRAME,
        cooperative_pose,
        scale=0.18,
    )
    cooperative_frame.visible = False

    def update_tool_targets_from_cooperative_frame() -> None:
        if not state["coordinate_mode"] or state["target_rel_pose"] is None:
            return
        cooperative_target = _pose_from_handle(cooperative_frame, device_cfg)
        tool_targets = _tool_poses_from_cooperative_pose(
            cooperative_target,
            state["target_rel_pose"],
        )
        for frame_name, pose in tool_targets.items():
            _set_handle_pose(viser_viz._control_frames[frame_name], pose)

    @cooperative_frame.on_update
    def _(_) -> None:
        update_tool_targets_from_cooperative_frame()

    def sync_targets_to_current_robot() -> None:
        tool_poses = _get_current_tool_poses(planner, state["current_state"])
        for frame_name, pose in tool_poses.items():
            _set_handle_pose(viser_viz._control_frames[frame_name], pose)
        _set_handle_pose(
            cooperative_frame,
            _cooperative_pose_from_tool_poses(
                tool_poses[LEFT_TOOL_FRAME],
                tool_poses[RIGHT_TOOL_FRAME],
            ),
        )

    def get_goal_poses() -> dict[str, Pose]:
        if state["coordinate_mode"]:
            update_tool_targets_from_cooperative_frame()
        return {
            frame_name: _pose_from_handle(viser_viz._control_frames[frame_name], device_cfg)
            for frame_name in [LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME]
        }

    def execute_trajectory(trajectory: JointState) -> None:
        traj = trajectory.squeeze(0)
        for i in range(traj.position.shape[-2]):
            if not state["is_moving"]:
                return
            waypoint = JointState.from_position(
                traj.position[0, i, :].unsqueeze(0),
                joint_names=traj.joint_names,
            )
            viser_viz.set_joint_state(waypoint.squeeze(0))
            time.sleep(0.02)
        state["current_state"] = JointState.from_position(
            traj.position[0, -1, :].unsqueeze(0),
            joint_names=traj.joint_names,
        )
        if state["coordinate_mode"]:
            sync_targets_to_current_robot()

    def on_coordinate(_) -> None:
        if state["is_moving"]:
            print("Coordinate toggle ignored while robot is moving.")
            return

        if state["coordinate_mode"]:
            state["coordinate_mode"] = False
            state["target_rel_pose"] = None
            _set_relative_pose_tracking(planner, enabled=False)
            cooperative_frame.visible = False
            print("Coordinate mode disabled; relative-pose cost is off.")
            return

        sync_targets_to_current_robot()
        tool_poses = _get_current_tool_poses(planner, state["current_state"])
        target_rel_pose = _relative_pose(
            tool_poses[LEFT_TOOL_FRAME],
            tool_poses[RIGHT_TOOL_FRAME],
        )
        state["target_rel_pose"] = target_rel_pose
        state["coordinate_mode"] = True
        _set_relative_pose_tracking(planner, enabled=True, target_rel_pose=target_rel_pose)
        cooperative_frame.visible = True
        update_tool_targets_from_cooperative_frame()
        print("Coordinate mode enabled; recorded current right-in-left relative pose.")

    def on_move(_) -> None:
        if state["is_moving"]:
            return

        def plan_and_execute() -> None:
            state["is_moving"] = True
            try:
                target_poses = get_goal_poses()
                active_js = planner.kinematics.get_active_js(state["current_state"].clone())
                result = planner.plan_pose(
                    GoalToolPose.from_poses(target_poses, num_goalset=1),
                    active_js,
                    use_implicit_goal=True,
                    max_attempts=3,
                )
                if result is not None and result.success.any():
                    print(f"Planning succeeded in {result.total_time:.3f}s")
                    interpolated_plan = result.get_interpolated_plan()
                    if state["coordinate_mode"] and state["target_rel_pose"] is not None:
                        if (
                            result.interpolated_metrics is not None
                            and result.interpolated_metrics.costs_and_constraints is not None
                        ):
                            constraint_names = (
                                result.interpolated_metrics.costs_and_constraints.constraints.names
                            )
                            print(f"Interpolated constraint metrics: {constraint_names}")
                        pos_max, pos_mean, rot_max, rot_mean = _trajectory_relative_pose_error(
                            planner,
                            interpolated_plan,
                            state["target_rel_pose"],
                        )
                        print(
                            "Relative pose over interpolated trajectory: "
                            f"pos max={pos_max:.4f} m, pos mean={pos_mean:.4f} m, "
                            f"rot max={rot_max:.4f} rad, rot mean={rot_mean:.4f} rad"
                        )
                        if (
                            args.reject_relative_violations
                            and (
                                pos_max > args.relative_position_tolerance
                                or rot_max > args.relative_orientation_tolerance
                            )
                        ):
                            print("Trajectory rejected: relative-pose constraint violated.")
                            return
                        if (
                            pos_max > args.relative_position_tolerance
                            or rot_max > args.relative_orientation_tolerance
                        ):
                            print("Relative-pose tolerance exceeded; executing trajectory anyway.")
                    execute_trajectory(interpolated_plan)
                else:
                    print("Motion planning failed")
                    if result is not None:
                        _print_plan_failure_details(result)
            finally:
                state["is_moving"] = False

        threading.Thread(target=plan_and_execute, daemon=True).start()

    def on_reset_targets(_) -> None:
        if state["is_moving"]:
            return
        sync_targets_to_current_robot()
        print("Target frames reset to current robot TCP poses.")

    coordinate_btn = server.gui.add_button("Coordinate", color="blue")
    coordinate_btn.on_click(on_coordinate)
    move_btn = server.gui.add_button("Move", color="green")
    move_btn.on_click(on_move)
    reset_btn = server.gui.add_button("Reset Targets", color="gray")
    reset_btn.on_click(on_reset_targets)

    print(f"\nViser running at http://localhost:{args.port}")
    print("  - Drag left/right target frames for ordinary dual-arm planning.")
    print("  - Click Coordinate to record the current relative pose and use one cooperative frame.")
    print("  - Drag the cooperative frame, then click Move to plan and execute.")
    print("  - Click Coordinate again to disable relative-pose tracking.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        planner.destroy()


if __name__ == "__main__":
    main()
