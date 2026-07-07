#!/usr/bin/env python3
"""Test dual-arm Panda config with obstacle avoidance.

Usage:
    python test_dual_panda.py test        # Run config + planning tests
    python test_dual_panda.py visualize   # Launch web visualizer at http://localhost:8082
"""

import sys
import time
import torch

from curobo.types import JointState, GoalToolPose
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.viewer import ViserVisualizer
from curobo.types import ContentPath


def get_planner():
    config = MotionPlannerCfg.create(
        robot="dual_panda.yml",
        scene_model="my_scene.yml",
    )
    return MotionPlanner(config)


def test_config_load():
    """Verify the dual_panda config loads correctly."""
    print("=== Testing dual_panda config load ===")
    planner = get_planner()
    print(f"  Joints ({len(planner.joint_names)}): {planner.joint_names}")
    print(f"  Tool frames: {planner.tool_frames}")
    print(f"  Default joint shape: {planner.default_joint_state.position.shape}")
    print("  ✓ Config loaded successfully")
    return planner


def test_ee_poses(planner):
    """Check end-effector positions at default pose."""
    print("\n=== End-effector FK ===")
    js = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )
    state = planner.kinematics.compute_kinematics(js)
    for i, tf in enumerate(planner.tool_frames):
        pos = state.tool_poses.position[0, 0, i, :].cpu().tolist()
        print(f"  {tf}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")


def test_planning(planner):
    """Plan dual-arm trajectory from default pose."""
    print("\n=== Testing dual-arm planning ===")
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    print("  ✓ Warmup done")

    js = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )

    # Build goal from current EE poses + small offset
    state = planner.kinematics.compute_kinematics(js)
    pos = state.tool_poses.position.unsqueeze(-2).clone()  # [B,H,L] → [B,H,L,G=1,3]
    quat = state.tool_poses.quaternion.unsqueeze(-2).clone()  # [B,H,L] → [B,H,L,G=1,4]

    # Move both arms forward by 10cm
    pos[0, 0, 0, 0, 0] += 0.10  # left:  +X
    pos[0, 0, 1, 0, 0] += 0.10  # right: +X

    goal = GoalToolPose(tool_frames=planner.tool_frames, position=pos, quaternion=quat)
    result = planner.plan_pose(goal, js)

    if result is not None and result.success.any():
        interp = result.get_interpolated_plan()
        print(f"  ✓ Planning succeeded! {interp.position.shape[-2]} waypoints")
    else:
        print("  ✗ Planning failed")


def test_obstacle_avoidance(planner):
    """Plan with active obstacles in the scene."""
    print("\n=== Testing obstacle avoidance ===")

    js = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )

    state = planner.kinematics.compute_kinematics(js)
    pos = state.tool_poses.position.unsqueeze(-2).clone()
    quat = state.tool_poses.quaternion.unsqueeze(-2).clone()

    # Left arm goes up+forward, right arm goes down+forward
    pos[0, 0, 0, 0, 0] += 0.15  # left:  +15cm X
    pos[0, 0, 0, 0, 2] += 0.10  # left:  +10cm Z
    pos[0, 0, 1, 0, 0] += 0.15  # right: +15cm X

    goal = GoalToolPose(tool_frames=planner.tool_frames, position=pos, quaternion=quat)
    result = planner.plan_pose(goal, js)

    if result is not None and result.success.any():
        interp = result.get_interpolated_plan()
        print(f"  ✓ Obstacle avoidance succeeded! {interp.position.shape[-2]} waypoints")
    else:
        print("  ✗ Planning failed (goal may be blocked by obstacles)")


def visualize_dual_arm():
    """Launch web visualizer with dual-arm setup."""
    print("\n=== Launching Web Visualizer ===")
    print("Open http://localhost:8082 in your browser")
    print("(Use VS Code Remote port forwarding: Ctrl+Shift+P → Forward a Port → 8082)")

    config = MotionPlannerCfg.create(
        robot="dual_panda.yml",
        scene_model="my_scene.yml",
    )

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file="dual_panda.yml"),
        connect_ip="0.0.0.0",
        connect_port=8082,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )

    scene_cfg = config.scene_collision_cfg.scene_model
    viser_viz.add_scene(scene_cfg, add_control_frames=True)
    viser_viz.set_joint_state(config.kinematics.default_joint_state)

    planner = MotionPlanner(config)
    print("Warming up planner (takes ~30s first time)...")
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    print(f"Ready! {len(planner.joint_names)} joints, 2 arms")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    if mode == "test":
        planner = test_config_load()
        test_ee_poses(planner)
        test_planning(planner)
        test_obstacle_avoidance(planner)
        print("\n=== All tests passed ===")
    elif mode == "visualize":
        visualize_dual_arm()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python test_dual_panda.py [test|visualize]")
