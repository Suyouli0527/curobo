# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive dual-arm motion planning with cuRobo v2 and Viser.

Run:
    python -m lego_stacking.main

Then open http://localhost:8080, drag the target frames to set goal poses
for each arm, and click "Move" to plan and execute a trajectory.
"""

from __future__ import annotations

import argparse
import threading
import time

import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState
from curobo.viewer import ViserVisualizer

from lego_stacking.scene_builder import create_scene_and_bricks


def main():
    parser = argparse.ArgumentParser(description="Interactive Dual-Arm Motion Planning with cuRobo v2")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--robot", type=str, default=None, help="Robot config file (default: from scene_config.json)")
    args = parser.parse_args()

    print("Loading scene config and layout...")
    scene, scene_cfg, bricks = create_scene_and_bricks()

    robot_config = args.robot if args.robot is not None else scene_cfg.robot_config
    print(f"Initializing MotionPlanner with robot={robot_config}...")
    config = MotionPlannerCfg.create(
        robot=robot_config,
        self_collision_check=True,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        collision_cache={"cuboid": 20, "mesh": 5},
    )
    planner = MotionPlanner(config)
    planner.update_world(scene)

    print("Warming up planner...")
    planner.warmup(enable_graph=False, num_warmup_iterations=5)

    print("Initializing Viser...")
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_config),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_robot_to_scene=True,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )

    # Visualize baseplate and bricks (collision-free, for reference only)
    viser_viz.add_scene(scene, add_control_frames=False)
    server = viser_viz._server
    for bid, brick in bricks.items():
        server.scene.add_box(
            name=f"/bricks/{bid}",
            dimensions=tuple(brick.dims),
            position=tuple(brick.world_pos),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            color=(100, 150, 200),
        )

    current_state = planner.default_joint_state.clone().unsqueeze(0)
    is_moving = False

    def execute_trajectory(trajectory):
        nonlocal current_state, is_moving
        traj = trajectory.squeeze(0)
        for i in range(traj.position.shape[-2]):
            if not is_moving:
                return
            waypoint = JointState.from_position(
                traj.position[0, i, :].unsqueeze(0),
                joint_names=traj.joint_names,
            )
            viser_viz.set_joint_state(waypoint.squeeze(0))
            time.sleep(0.02)
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
            result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
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
