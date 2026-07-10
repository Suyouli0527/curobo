"""双臂 Panda 避障可视化 — 入口脚本.

Usage:
    python -m run.main                                    # 默认场景 1
    python -m run.main --scene myscene_2.yml              # 指定场景
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from run.scene import load_scene_yaml, setup_viser, render_scene_obstacles, create_control_frames
from run.controller import DualArmMPCController
from run.gripper import Gripper


def main():
    parser = argparse.ArgumentParser(description="双臂 Panda reactive MPC 避障")
    parser.add_argument("--robot", default="dual_franka_with_camera.yml")
    parser.add_argument("--scene", default="myscene_1.yml")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print(f"Loading scene: {args.scene}")
    scene_dict = load_scene_yaml(args.scene)

    viser_viz, server = setup_viser(args.robot, args.port)
    obstacle_info = render_scene_obstacles(server, scene_dict)

    controller = DualArmMPCController(robot_file=args.robot, scene_dict=scene_dict)
    gripper = Gripper(obstacle_info)

    (left_pos, left_wxyz), (right_pos, right_wxyz) = controller.compute_ee_poses()
    left_frame, right_frame = create_control_frames(
        server, (left_pos, left_wxyz), (right_pos, right_wxyz)
    )
    controller.set_initial_goal((left_pos, left_wxyz), (right_pos, right_wxyz))
    viser_viz.set_joint_state(controller.current_state.squeeze(0))

    pending_coord_toggle = False
    pending_grasp = False
    pending_release = False

    coord_btn = server.gui.add_button("Coordinate", color=(128, 128, 128))
    grasp_btn = server.gui.add_button("Grasp", color="green")
    release_btn = server.gui.add_button("Release", color="red")

    def on_coordinate(_):
        nonlocal pending_coord_toggle
        pending_coord_toggle = True
    def on_grasp(_):
        nonlocal pending_grasp
        if gripper.attached_name is None:
            pending_grasp = True
    def on_release(_):
        nonlocal pending_release
        if gripper.attached_name is not None:
            pending_release = True

    coord_btn.on_click(on_coordinate)
    grasp_btn.on_click(on_grasp)
    release_btn.on_click(on_release)

    print(f"\n✓ Reactive MPC 就绪! http://localhost:{args.port}")
    print(f"  机器人: {args.robot}  |  场景: {args.scene}")
    print(f"  Joints: {len(controller.joint_names)} | Tool frames: {controller.tool_frames}")
    print("  拖拽控制框 → 实时追踪  |  Coordinate → 协同")
    print("  Grasp → 抓取最近障碍物  |  Release → 释放")
    print("  Press Ctrl+C to exit\n")

    step = 0
    left_prev_pos = left_pos.copy()
    right_prev_pos = right_pos.copy()
    left_prev_wxyz = left_wxyz.copy()
    right_prev_wxyz = right_wxyz.copy()

    try:
        while True:
            if pending_coord_toggle:
                pending_coord_toggle = False
                controller.toggle_coordination()
                coord_btn.color = (
                    (255, 165, 0) if controller.coordinated else (128, 128, 128)
                )
                step = 0
                continue

            if pending_grasp:
                pending_grasp = False
                (l, _), (r, _) = controller.compute_ee_poses()
                gripper.grasp((l + r) / 2, (l + r) / 2)
            if pending_release:
                pending_release = False
                gripper.release()

            pose_changed = False
            left_moved = (
                np.any(left_frame.position != left_prev_pos)
                or np.any(left_frame.wxyz != left_prev_wxyz)
            )
            right_moved = (
                np.any(right_frame.position != right_prev_pos)
                or np.any(right_frame.wxyz != right_prev_wxyz)
            )

            if not controller.coordinated:
                if left_moved:
                    left_prev_pos = left_frame.position.copy()
                    left_prev_wxyz = left_frame.wxyz.copy()
                    pose_changed = True
                if right_moved:
                    right_prev_pos = right_frame.position.copy()
                    right_prev_wxyz = right_frame.wxyz.copy()
                    pose_changed = True
                if pose_changed:
                    controller.update_goal(
                        (left_frame.position, left_frame.wxyz),
                        (right_frame.position, right_frame.wxyz),
                    )
            else:
                if left_moved and not right_moved:
                    left_prev_pos = left_frame.position.copy()
                    left_prev_wxyz = left_frame.wxyz.copy()
                    rp, rq = controller.sync_frame_left_to_right(left_prev_pos, left_prev_wxyz)
                    right_frame.position = rp
                    right_frame.wxyz = rq
                    right_prev_pos = rp.copy()
                    right_prev_wxyz = rq.copy()
                    pose_changed = True
                elif right_moved and not left_moved:
                    right_prev_pos = right_frame.position.copy()
                    right_prev_wxyz = right_frame.wxyz.copy()
                    lp, lq = controller.sync_frame_right_to_left(right_prev_pos, right_prev_wxyz)
                    left_frame.position = lp
                    left_frame.wxyz = lq
                    left_prev_pos = lp.copy()
                    left_prev_wxyz = lq.copy()
                    pose_changed = True
                elif left_moved and right_moved:
                    left_prev_pos = left_frame.position.copy()
                    left_prev_wxyz = left_frame.wxyz.copy()
                    right_prev_pos = right_frame.position.copy()
                    right_prev_wxyz = right_frame.wxyz.copy()
                    pose_changed = True
                if pose_changed:
                    controller.update_goal(
                        (left_prev_pos, left_prev_wxyz),
                        (right_prev_pos, right_prev_wxyz),
                    )

            if controller.step():
                viser_viz.set_joint_state(controller.current_state.squeeze(0))
                if gripper.attached_name is not None:
                    (l, _), (r, _) = controller.compute_ee_poses()
                    gripper.update((l + r) / 2)
                step += 1
                if step % 100 == 0:
                    print(f"  Step {step}")

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        controller.destroy()


if __name__ == "__main__":
    main()
