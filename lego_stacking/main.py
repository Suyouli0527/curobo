# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive dual-arm motion planning with cuRobo v2 and Viser.

Run:
    python -m lego_stacking.main

Then open http://localhost:8080, drag the target frames to set goal poses
for each arm, and click "Move" to plan and execute a trajectory.

Features:
- Optional manipulability-maximizing IK (selects the IK seed with highest
  Yoshikawa manipulability across both arms).
- Optional real-time manipulability ellipsoid visualization in Viser.
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import numpy as np
import torch
import trimesh

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState
from curobo.viewer import ViserVisualizer

from lego_stacking.scene_builder import apply_robot_base_poses_to_config, create_scene_and_bricks


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------
def _quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


# ---------------------------------------------------------------------------
# Manipulability computation
# ---------------------------------------------------------------------------
def compute_manipulability_from_state(
    kin_state: KinematicsState,
    kin_jac: Kinematics,
    tool_frames: list[str],
) -> tuple[torch.Tensor | None, list[dict] | None]:
    """Compute Yoshikawa manipulability and ellipsoid SVD data.

    Args:
        kin_state: Kinematics state with ``tool_jacobians`` populated.
        kin_jac: Kinematics instance (for tool-frame indexing).
        tool_frames: Tool frame names to evaluate.

    Returns:
        - scores: ``[batch, horizon, num_tool_frames]`` or None if no jacobians.
        - ellipsoids: List of dicts with SVD data or None.
    """
    jacobians = kin_state.tool_jacobians
    if jacobians is None:
        return None, None

    batch, horizon = jacobians.shape[0], jacobians.shape[1]
    scores: list[torch.Tensor] = []
    ellipsoids: list[dict] = []

    for tf in tool_frames:
        tool_idx = kin_jac.tool_frames.index(tf)
        J_pos = jacobians[:, :, tool_idx, :3, :]  # [B, H, 3, dof]
        J_flat = J_pos.reshape(batch * horizon, 3, J_pos.shape[-1])
        U, S, _ = torch.linalg.svd(J_flat, full_matrices=False)
        U = U.reshape(batch, horizon, 3, 3)
        S = S.reshape(batch, horizon, 3)
        w = torch.prod(S, dim=-1)  # [B, H]
        scores.append(w)
        ellipsoids.append({"U": U, "S": S, "tool_frame": tf})

    return torch.stack(scores, dim=-1), ellipsoids


def build_manipulability_ellipsoid_mesh(
    U: np.ndarray, S: np.ndarray, scale: float = 0.05
) -> trimesh.Trimesh:
    """Build a trimesh ellipsoid from SVD parameters of the position Jacobian."""
    S = np.clip(S, 1e-6, None)
    S_norm = S / (S.max() + 1e-6)
    S_vis = np.clip(S_norm * scale, 1e-4, scale * 2)

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    sphere.vertices *= S_vis
    sphere.vertices = sphere.vertices @ U.T
    return sphere


def select_best_ik_seed_by_manipulability(
    ik_result,
    kin_jac: Kinematics,
    tool_frames: list[str],
) -> tuple[torch.Tensor | None, int | None, int | None, torch.Tensor | None]:
    """Select the IK seed with highest average manipulability.

    Returns:
        - best_seed: ``[1, dof]`` tensor or ``None`` if no success.
        - best_idx: Original seed index or ``None``.
        - best_local: Index within successful seeds or ``None``.
        - all_scores: ``[num_success, num_tool_frames]`` or ``None``.
    """
    batch_idx = 0
    success_mask = ik_result.success[batch_idx]
    if success_mask.sum() == 0:
        return None, None, None, None

    seed_configs = ik_result.solution[batch_idx]
    success_configs = seed_configs[success_mask]

    js_batch = JointState.from_position(success_configs, joint_names=kin_jac.joint_names)
    kin_state = kin_jac.compute_kinematics(js_batch)
    scores, _ = compute_manipulability_from_state(kin_state, kin_jac, tool_frames)
    if scores is None:
        return None, None, None, None

    avg_scores = scores[:, 0, :].mean(dim=-1)
    best_local = int(avg_scores.argmax().item())
    success_indices = success_mask.nonzero(as_tuple=True)[0]
    best_idx = int(success_indices[best_local].item())

    return success_configs[best_local].unsqueeze(0), best_idx, best_local, scores[:, 0, :]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Interactive Dual-Arm Motion Planning with cuRobo v2")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--robot", type=str, default=None, help="Robot config file")
    args = parser.parse_args()

    print("Loading scene config and layout...")
    scene, scene_cfg, bricks = create_scene_and_bricks()

    robot_config = args.robot if args.robot is not None else scene_cfg.robot_config
    if scene_cfg.robot_base_poses is not None:
        print("Applying custom robot base poses from scene config...")
        robot_config = apply_robot_base_poses_to_config(robot_config, scene_cfg.robot_base_poses)

    print(f"Initializing MotionPlanner with robot={scene_cfg.robot_config}...")
    config = MotionPlannerCfg.create(
        robot=robot_config,
        scene_model=scene,
        self_collision_check=True,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        collision_cache={"cuboid": 20, "mesh": 5},
    )
    planner = MotionPlanner(config)

    print("Warming up planner...")
    planner.warmup(enable_graph=False, num_warmup_iterations=5)

    # Auxiliary kinematics with Jacobian computation for manipulability.
    print("Initializing manipulability kinematics...")
    kin_with_jac = Kinematics(
        planner.kinematics.config,
        compute_jacobian=True,
        compute_spheres=False,
        compute_com=False,
    )
    # Warmup to JIT-compile the jacobian kernel.
    _ = kin_with_jac.compute_kinematics(planner.default_joint_state)

    print("Initializing Viser...")
    viser_content = (
        robot_config if isinstance(robot_config, dict)
        else ContentPath(robot_config_file=robot_config)
    )
    viser_viz = ViserVisualizer(
        content_path=viser_content,
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_robot_to_scene=True,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )

    viser_viz.add_scene(scene, add_control_frames=False)
    server = viser_viz._server
    for bid, brick in bricks.items():
        server.scene.add_box(
            name=f"/bricks/{bid}",
            dimensions=tuple(brick.dims),
            position=tuple(brick.world_pos),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            color=(255, 0, 0),
        )

    current_state = planner.default_joint_state.clone().unsqueeze(0)
    is_moving = False

    # ---- GUI controls for manipulability ----
    enable_manip_opt = server.gui.add_checkbox(
        "Maximize Manipulability (IK)", initial_value=False
    )
    enable_manip_vis = server.gui.add_checkbox(
        "Show Manipulability Ellipsoids", initial_value=False
    )
    ellipsoid_scale = server.gui.add_slider(
        "Ellipsoid Scale", min=0.01, max=0.2, step=0.01, initial_value=0.05
    )

    _ellipsoid_frame_names: list[str] = []
    _ellipsoid_counter = 0

    def clear_ellipsoids():
        nonlocal _ellipsoid_counter
        for name in _ellipsoid_frame_names:
            try:
                server.scene.remove_frame(name)
            except Exception:
                pass
        _ellipsoid_frame_names.clear()
        _ellipsoid_counter += 1

    def update_ellipsoids(joint_state: JointState):
        nonlocal _ellipsoid_counter
        clear_ellipsoids()
        if not enable_manip_vis.value:
            return

        try:
            js = joint_state.clone()
            expected = kin_with_jac.joint_names
            if js.joint_names is not None and list(js.joint_names) != list(expected):
                js.reindex(expected)
            kin_state = kin_with_jac.compute_kinematics(js)
        except Exception as e:
            print(f"[manipulability] compute_kinematics skipped: {e}")
            return

        _, ellipsoids = compute_manipulability_from_state(
            kin_state, kin_with_jac, planner.kinematics.tool_frames
        )
        if ellipsoids is None:
            return

        scale = ellipsoid_scale.value
        frame_prefix = f"/manip_ellipsoid_{_ellipsoid_counter}"

        for data in ellipsoids:
            tf = data["tool_frame"]
            pose = kin_state.tool_poses[tf]
            U = data["U"][0, 0].cpu().numpy()
            S = data["S"][0, 0].cpu().numpy()

            mesh = build_manipulability_ellipsoid_mesh(U, S, scale=scale)
            pos = pose.position.cpu().squeeze().numpy()
            quat = pose.quaternion.cpu().squeeze().numpy()
            R = _quat_to_rot_matrix(quat)
            mesh.vertices = mesh.vertices @ R.T + pos

            # Color by manipulability (green=high, red=low)
            w = float(np.prod(S))
            w_norm = min(1.0, w / 0.01)
            rgba = [
                int((1.0 - w_norm) * 255),
                int(w_norm * 255),
                50,
                200,
            ]
            vc = np.tile(rgba, (len(mesh.vertices), 1)).astype(np.uint8)
            mesh.visual.vertex_colors = vc

            name = f"{frame_prefix}/{tf}"
            server.scene.add_mesh_trimesh(name=name, mesh=mesh)
            _ellipsoid_frame_names.append(name)

    def execute_trajectory(trajectory):
        nonlocal current_state, is_moving
        traj = trajectory.squeeze(0)
        for i in range(traj.position.shape[-2]):
            if not is_moving:
                clear_ellipsoids()
                return
            waypoint = JointState.from_position(
                traj.position[0, i, :].unsqueeze(0),
                joint_names=traj.joint_names,
            )
            viser_viz.set_joint_state(waypoint.squeeze(0))
            if enable_manip_vis.value:
                update_ellipsoids(waypoint)
            time.sleep(0.02)
        clear_ellipsoids()
        current_state = JointState.from_position(
            traj.position[0, -1, :].unsqueeze(0),
            joint_names=traj.joint_names,
        )

    def on_move(_):
        nonlocal is_moving
        if is_moving:
            return

        def plan_and_execute():
            nonlocal is_moving
            is_moving = True
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())
            goal = GoalToolPose.from_poses(target_poses, num_goalset=1)

            if enable_manip_opt.value:
                ik_result = planner.ik_solver.solve_pose(
                    goal,
                    current_state=active_js,
                    return_seeds=32,
                )
                best_seed, best_idx, best_local, all_scores = select_best_ik_seed_by_manipulability(
                    ik_result, kin_with_jac, planner.kinematics.tool_frames
                )
                if best_seed is None:
                    print("Motion planning failed: no successful IK seed")
                    is_moving = False
                    return

                avg_score = float(all_scores[best_local].mean().item())
                print(f"Selected IK seed {best_idx} (avg manipulability={avg_score:.6f})")

                num_traj_seeds = planner.trajopt_solver.config.num_seeds
                seed_config = best_seed.unsqueeze(0).repeat(1, num_traj_seeds, 1)
                result = planner.trajopt_solver.solve_pose(
                    goal,
                    active_js,
                    seed_config=seed_config,
                    use_implicit_goal=True,
                )
            else:
                result = planner.plan_pose(
                    goal,
                    active_js,
                    use_implicit_goal=True,
                    max_attempts=3,
                )

            if result is not None and result.success.any():
                interp = result.get_interpolated_plan()
                execute_trajectory(interp)
            else:
                print("Motion planning failed")
            is_moving = False

        threading.Thread(target=plan_and_execute, daemon=True).start()

    move_btn = server.gui.add_button("Move", color="green")
    move_btn.on_click(on_move)

    print(f"\nViser running at http://localhost:{args.port}")
    print("  - Drag the target frames to set goal poses for each arm")
    print("  - Click 'Move' to plan and execute a trajectory")
    print("  - Toggle 'Maximize Manipulability (IK)' to optimize for dexterity")
    print("  - Toggle 'Show Manipulability Ellipsoids' for real-time visualization")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        planner.destroy()
        # Clean up temp URDF generated by apply_robot_base_poses_to_config.
        if isinstance(robot_config, dict):
            temp_urdf = robot_config.get("robot_cfg", {}).get("kinematics", {}).get("urdf_path")
            if temp_urdf is not None and os.path.exists(temp_urdf):
                os.remove(temp_urdf)


if __name__ == "__main__":
    main()
