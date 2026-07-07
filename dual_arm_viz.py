#!/usr/bin/env python3
"""双臂 Panda 避障可视化 — cuRobo reactive MPC + Viser

完全参照 curobo.examples.getting_started.reactive_control 的 interactive_mpc_example 模式:
  - 拖拽控制框即可实时追踪，无需点击按钮
  - 每帧自动检测位姿变化 + warm-start 重优化
  - 所有 CUDA 操作在主线程完成，避免 stream capture 冲突

Usage:
    python dual_arm_viz.py                                    # 默认场景 1
    python dual_arm_viz.py --scene myscene_2.yml              # 指定场景
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer
from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg

CUROBO_WS = Path(__file__).parent
SCENE_CONFIG_DIR = CUROBO_WS / "curobo" / "content" / "configs" / "scene"


def load_scene(scene_name: str) -> dict:
    path = SCENE_CONFIG_DIR / scene_name
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    path = CUROBO_WS / scene_name
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    print(f"✗ Scene file not found: {scene_name}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="双臂 Panda reactive MPC 避障")
    parser.add_argument("--robot", default="dual_franka_with_camera.yml")
    parser.add_argument("--scene", default="myscene_1.yml")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print(f"Loading scene: {args.scene}")
    scene_dict = load_scene(args.scene)

    # ── 1. Viser 初始化 ──────────────────────────────────
    print("Initializing Viser...")
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=args.robot),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_robot_to_scene=True,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )
    server = viser_viz._server  # noqa: SLF001 (official pattern)

    # ── 2. MPC 配置 ──────────────────────────────────────
    print(f"Initializing MPC with robot={args.robot}...")
    config = ModelPredictiveControlCfg.create(
        robot=args.robot,
        scene_model=scene_dict,
        self_collision_check=True,
        collision_cache={"cuboid": 50, "mesh": 10},
        optimization_dt=0.025,
        use_cuda_graph=True,
        cold_start_optimization_num_iters=300,
        warm_start_optimization_num_iters=200,
    )
    mpc = ModelPredictiveControl(config)

    # ── 3. 场景障碍物 (从 YAML 渲染) ─────────────────────
    _color_map = {
        "seat": (160, 100, 50), "leg_": (180, 130, 70), "back_": (140, 90, 40),
        "tray_": (120, 120, 130), "cup": (200, 200, 210), "bottle": (180, 200, 220),
        "long_": (140, 110, 70), "beam_": (100, 100, 110), "divider_": (90, 90, 100),
        "back_wall": (80, 80, 90), "pillar_": (150, 150, 160), "p_": (150, 150, 160),
        "inner_": (140, 140, 150), "rod_": (180, 130, 70),
        "screw": (100, 100, 110), "nut": (100, 100, 110), "socket": (80, 80, 90),
        "overhead": (130, 130, 140), "clutter_": (170, 170, 100),
        "obs_": (180, 80, 80), "distractor": (170, 100, 100),
    }

    def _get_color(name):
        for pfx, c in _color_map.items():
            if name.startswith(pfx):
                return c
        return (180, 180, 50)

    from curobo.scene import Scene
    scene = Scene.create(scene_dict)
    if hasattr(scene, 'cuboid') and scene.cuboid:
        for obs in scene.cuboid:
            c = _get_color(obs.name)
            server.scene.add_box(
                f"/obstacles/{obs.name}",
                dimensions=np.array(obs.dims),
                position=np.array(obs.pose[:3]),
                wxyz=np.array(obs.pose[3:]),
                color=np.array(c, dtype=np.uint8),
            )
    if hasattr(scene, 'sphere') and scene.sphere:
        for obs in scene.sphere:
            c = _get_color(obs.name)
            server.scene.add_icosphere(
                f"/obstacles/{obs.name}",
                radius=obs.radius,
                position=np.array(obs.pose[:3]),
                color=np.array(c, dtype=np.uint8),
            )

    # ── 4. 双臂协调约束 (默认关闭) ────────────────────────
    rel_pose_cfg = RelativePoseCostCfg(
        weight=torch.tensor([50000.0, 5000.0]),
        primary_tool_frame="panda_left_hand_tcp",
        secondary_tool_frame="panda_right_hand_tcp",
        position_tolerance=0.01,
        orientation_tolerance=0.05,
    )
    core = mpc.config.core_cfg
    for rc in core.optimizer_rollout_configs:
        if rc.cost_cfg is not None:
            rc.cost_cfg.relative_pose_cfg = rel_pose_cfg
        if rc.constraint_cfg is not None:
            rc.constraint_cfg.relative_pose_cfg = rel_pose_cfg
    mr = core.metrics_rollout_config
    if mr is not None:
        if mr.cost_cfg is not None:
            mr.cost_cfg.relative_pose_cfg = rel_pose_cfg
        if mr.constraint_cfg is not None:
            mr.constraint_cfg.relative_pose_cfg = rel_pose_cfg
    for rollout in mpc.core.get_all_rollout_instances():
        for mgr in rollout._cost_manager_list:
            if mgr.has_cost("relative_pose"):
                mgr.get_cost("relative_pose").disable_cost()

    # ── 5. 创建初始状态 + MPC setup ──────────────────────
    current_state = JointState.from_position(
        mpc.default_joint_position.clone().unsqueeze(0),
        joint_names=mpc.joint_names,
    )
    current_state.velocity = torch.zeros_like(current_state.position)
    current_state.acceleration = torch.zeros_like(current_state.position)

    print("Setting up MPC...")
    mpc.setup(current_state)
    print("  MPC setup complete")

    # ── 6. 初始目标 = 当前 FK 位姿 (机器人保持静止) ──────
    kin_result = mpc.compute_kinematics(current_state)
    target_link_poses = kin_result.tool_poses.to_dict()
    mpc.update_goal_tool_poses(
        GoalToolPose.from_poses(
            target_link_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1,
        ),
        run_ik=False,  # 初始目标 = 当前位置，无需 IK
    )
    print(f"  Initial goal: {list(target_link_poses.keys())}")
    viser_viz.set_joint_state(current_state.squeeze(0))

    # ── 7. 协调按钮 + 交互控件 ────────────────────────────
    coordinated = False
    pending_coord_toggle = False

    coord_btn = server.gui.add_button("Coordinate", color=(128, 128, 128))

    def on_coordinate(_):
        nonlocal pending_coord_toggle
        pending_coord_toggle = True

    coord_btn.on_click(on_coordinate)

    # ── 8. 主循环 (完全参照 interactive_mpc_example) ──────
    print(f"\n✓ Reactive MPC 就绪! http://localhost:{args.port}")
    print(f"  机器人: {args.robot}  |  场景: {args.scene}")
    print(f"  Joints: {len(mpc.joint_names)}  |  Tool frames: {mpc.tool_frames}")
    print("  拖拽绿色控制框 → 机器人实时追踪")
    print("  点击 Coordinate → 锁定双臂相对位姿 (协同搬运)")
    print("  Press Ctrl+C to exit\n")

    previous_target_poses = None
    pose_changed = False
    step = 0

    try:
        while True:
            # ── 检测控制框位姿变化 (参照 official pattern) ──
            target_poses = viser_viz.get_control_frame_pose()

            if previous_target_poses is None:
                previous_target_poses = target_poses
            else:
                for frame_name in target_poses:
                    if target_poses[frame_name] != previous_target_poses[frame_name]:
                        previous_target_poses = {
                            k: v.clone() for k, v in target_poses.items()
                        }
                        pose_changed = True
                        break

            # ── 位姿变化 → 更新目标 ─────────────────────────
            if pose_changed:
                # 去掉 "target_" 前缀 → tool frame 名
                target_link_poses = {
                    k.replace("target_", ""): v for k, v in target_poses.items()
                }
                print(f"[mpc] New targets: {list(target_link_poses.keys())}")
                mpc.update_goal_tool_poses(
                    GoalToolPose.from_poses(
                        target_link_poses,
                        ordered_tool_frames=mpc.tool_frames,
                        num_goalset=1,
                    ),
                    run_ik=False,  # 参照官方: 不跑 IK, MPC warm-start 直接追踪
                )
                pose_changed = False
                step = 0

            # ── 协调模式切换 ────────────────────────────────
            if pending_coord_toggle:
                pending_coord_toggle = False
                coordinated = not coordinated
                if coordinated:
                    from curobo.types import Pose as CuroboPose
                    kin = mpc.compute_kinematics(current_state)
                    tp = kin.tool_poses
                    rel_pos = (
                        tp["panda_right_hand_tcp"].position
                        - tp["panda_left_hand_tcp"].position
                    )
                    rel = CuroboPose(
                        position=rel_pos.clone(),
                        quaternion=tp["panda_right_hand_tcp"].quaternion.clone(),
                    )
                    for rollout in mpc.core.get_all_rollout_instances():
                        for mgr in rollout._cost_manager_list:
                            if mgr.has_cost("relative_pose"):
                                cost = mgr.get_cost("relative_pose")
                                cost.target_relative_pose = rel.clone()
                                cost.config.target_relative_pose = rel.clone()
                                cost.enable_cost()
                    print(f"[coord] ON — locked rel pos={rel_pos.squeeze().tolist()}")
                else:
                    for rollout in mpc.core.get_all_rollout_instances():
                        for mgr in rollout._cost_manager_list:
                            if mgr.has_cost("relative_pose"):
                                mgr.get_cost("relative_pose").disable_cost()
                    print("[coord] OFF")
                coord_btn.color = (255, 165, 0) if coordinated else (128, 128, 128)

            # ── Reactive MPC step ───────────────────────────
            mpc_result = mpc.optimize_action_sequence(current_state)

            if (
                mpc_result is not None
                and mpc_result.action_sequence is not None
                and mpc_result.action_sequence.position.shape[1] > 0
            ):
                next_position = mpc_result.action_sequence.position[:, -1, :]
                current_state = JointState.from_position(
                    next_position.clone(),
                    joint_names=mpc.joint_names,
                )
                current_state.velocity = mpc_result.action_sequence.velocity[:, -1, :]
                current_state.acceleration = mpc_result.action_sequence.acceleration[:, -1, :]
                viser_viz.set_joint_state(current_state.squeeze(0))

                step += 1
                if step % 100 == 0:
                    pos_err = mpc_result.position_error
                    err_str = f"{pos_err.item():.4f}" if pos_err is not None else "N/A"
                    print(f"  Step {step}: position error = {err_str}")

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        mpc.destroy()


if __name__ == "__main__":
    main()
