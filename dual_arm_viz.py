#!/usr/bin/env python3
"""双臂 Panda 避障可视化 — cuRobo reactive MPC + Viser

参照 curobo.examples.getting_started.reactive_control 的 interactive_mpc_example 模式:
  - 拖拽控制框即可实时追踪
  - 点击 Coordinate → 拖动任一侧框，另一侧自动跟随保持固定相对位姿
  - 所有 CUDA 操作在主线程完成

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
from curobo._src.geom.quaternion import quat_multiply

# ═══════════════════════════════════════════════════════════════
# 辅助函数层 — 数学工具 + IO 工具
# ═══════════════════════════════════════════════════════════════

# -- 四元数 (torch) --


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate (inverse for unit quaternions). q = [w, x, y, z]."""
    c = q.clone()
    c[..., 1:] *= -1
    return c


def _quat_rotate_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by unit quaternion q: v_rot = q * v * q^{-1}."""
    q_w, q_x, q_y, q_z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    v_x, v_y, v_z = v[..., 0], v[..., 1], v[..., 2]
    t_x = 2.0 * (q_y * v_z - q_z * v_y)
    t_y = 2.0 * (q_z * v_x - q_x * v_z)
    t_z = 2.0 * (q_x * v_y - q_y * v_x)
    return torch.stack([
        v_x + q_w * t_x + (q_y * t_z - q_z * t_y),
        v_y + q_w * t_y + (q_z * t_x - q_x * t_z),
        v_z + q_w * t_z + (q_x * t_y - q_y * t_x),
    ], dim=-1)


# -- 四元数 (numpy) --


def _quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + w2 * x1 + y1 * z2 - y2 * z1,
        w1 * y2 + w2 * y1 + z1 * x2 - z2 * x1,
        w1 * z2 + w2 * z1 + x1 * y2 - x2 * y1,
    ], dtype=np.float64)


def _quat_rotate_vector_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    v_x, v_y, v_z = v[0], v[1], v[2]
    t_x = 2.0 * (q_y * v_z - q_z * v_y)
    t_y = 2.0 * (q_z * v_x - q_x * v_z)
    t_z = 2.0 * (q_x * v_y - q_y * v_x)
    return np.array([
        v_x + q_w * t_x + (q_y * t_z - q_z * t_y),
        v_y + q_w * t_y + (q_z * t_x - q_x * t_z),
        v_z + q_w * t_z + (q_x * t_y - q_y * t_x),
    ], dtype=np.float64)


# -- 位姿转换 --


def _pose_to_numpy(position: torch.Tensor, quaternion: torch.Tensor):
    return (
        position.detach().cpu().squeeze().numpy().astype(np.float64),
        quaternion.detach().cpu().squeeze().numpy().astype(np.float64),
    )


def _numpy_to_pose(pos: np.ndarray, wxyz: np.ndarray, device: torch.device) -> Pose:
    return Pose(
        position=torch.tensor(pos, dtype=torch.float32, device=device).view(1, 1, 1, 3),
        quaternion=torch.tensor(wxyz, dtype=torch.float32, device=device).view(1, 1, 1, 4),
    )


# -- 场景加载 --

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


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════


def main():
    # ============================================================
    #  参数解析
    # ============================================================
    parser = argparse.ArgumentParser(description="双臂 Panda reactive MPC 避障")
    parser.add_argument("--robot", default="dual_franka_with_camera.yml")
    parser.add_argument("--scene", default="myscene_1.yml")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print(f"Loading scene: {args.scene}")
    scene_dict = load_scene(args.scene)

    # ============================================================
    #  可视化层 — Viser, 场景障碍物, 控制框
    # ============================================================

    # --- Viser 初始化 ---
    print("Initializing Viser...")
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=args.robot),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_robot_to_scene=True,
        add_control_frames=False,
        visualize_robot_spheres=False,
    )
    server = viser_viz._server  # noqa: SLF001

    # --- 场景障碍物渲染 ---
    _COLOR_MAP = {
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
        for pfx, c in _COLOR_MAP.items():
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

    # ============================================================
    #  控制层 — MPC, 约束, 状态, 控制框, 按钮, 主循环
    # ============================================================

    # --- MPC 初始化 ---
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

    # 禁用内置 IK solver 的 CUDA graph (避免与 MPC graph 冲突)
    try:
        for key in list(mpc.ik_solver.__dict__.keys()):
            if "seed" in key and hasattr(getattr(mpc.ik_solver, key), "config"):
                getattr(mpc.ik_solver, key).config.use_cuda_graph = False
    except Exception:
        pass

    # --- 协同约束配置 ---
    # soft 路径: 连续梯度引导
    rel_pose_cfg = RelativePoseCostCfg(
        weight=torch.tensor([50000.0, 5000.0]),
        primary_tool_frame="panda_left_hand_tcp",
        secondary_tool_frame="panda_right_hand_tcp",
        position_tolerance=0.01, orientation_tolerance=0.05,
    )
    # hard 路径: convert_to_binary=True, 违反即不可行
    rel_pose_hard_cfg = RelativePoseCostCfg(
        weight=torch.tensor([500000.0, 50000.0]),
        primary_tool_frame="panda_left_hand_tcp",
        secondary_tool_frame="panda_right_hand_tcp",
        position_tolerance=0.01, orientation_tolerance=0.05,
        convert_to_binary=True,
    )
    core = mpc.config.core_cfg
    for rc in core.optimizer_rollout_configs:
        if rc.cost_cfg is not None:
            rc.cost_cfg.relative_pose_cfg = rel_pose_cfg
        if rc.constraint_cfg is not None:
            rc.constraint_cfg.relative_pose_cfg = rel_pose_hard_cfg
    mr = core.metrics_rollout_config
    if mr is not None:
        if mr.cost_cfg is not None:
            mr.cost_cfg.relative_pose_cfg = rel_pose_cfg
        if mr.constraint_cfg is not None:
            mr.constraint_cfg.relative_pose_cfg = rel_pose_hard_cfg
    for rollout in mpc.core.get_all_rollout_instances():
        for mgr in rollout._cost_manager_list:
            if mgr.has_cost("relative_pose"):
                mgr.get_cost("relative_pose").disable_cost()

    # --- 初始状态 + MPC setup ---
    current_state = JointState.from_position(
        mpc.default_joint_position.clone().unsqueeze(0),
        joint_names=mpc.joint_names,
    )
    current_state.velocity = torch.zeros_like(current_state.position)
    current_state.acceleration = torch.zeros_like(current_state.position)

    print("Setting up MPC...")
    mpc.setup(current_state)
    print("  MPC setup complete")

    # --- 控制框创建 ---
    LEFT_TF = "panda_left_hand_tcp"
    RIGHT_TF = "panda_right_hand_tcp"

    kin_result = mpc.compute_kinematics(current_state)
    left_ee_pos, left_ee_wxyz = _pose_to_numpy(
        kin_result.tool_poses[LEFT_TF].position,
        kin_result.tool_poses[LEFT_TF].quaternion,
    )
    right_ee_pos, right_ee_wxyz = _pose_to_numpy(
        kin_result.tool_poses[RIGHT_TF].position,
        kin_result.tool_poses[RIGHT_TF].quaternion,
    )

    left_frame = server.scene.add_transform_controls(
        name="/target/left", scale=0.15,
        position=left_ee_pos, wxyz=left_ee_wxyz,
    )
    right_frame = server.scene.add_transform_controls(
        name="/target/right", scale=0.15,
        position=right_ee_pos, wxyz=right_ee_wxyz,
    )

    # 初始目标 = 当前 FK (机器人保持静止)
    mpc.update_goal_tool_poses(
        GoalToolPose.from_poses(
            {
                LEFT_TF: _numpy_to_pose(
                    left_ee_pos, left_ee_wxyz, current_state.position.device
                ),
                RIGHT_TF: _numpy_to_pose(
                    right_ee_pos, right_ee_wxyz, current_state.position.device
                ),
            },
            ordered_tool_frames=mpc.tool_frames,
            num_goalset=1,
        ),
        run_ik=False,
    )
    viser_viz.set_joint_state(current_state.squeeze(0))

    # --- 按钮回调 ---
    coordinated = False
    pending_coord_toggle = False
    coord_delta_pos_local = None
    coord_delta_quat_local = None

    coord_btn = server.gui.add_button("Coordinate", color=(128, 128, 128))

    def on_coordinate(_):
        nonlocal pending_coord_toggle
        pending_coord_toggle = True

    coord_btn.on_click(on_coordinate)

    # ============================================================
    #  主循环
    # ============================================================
    print(f"\n✓ Reactive MPC 就绪! http://localhost:{args.port}")
    print(f"  机器人: {args.robot}  |  场景: {args.scene}")
    print(f"  Joints: {len(mpc.joint_names)}  |  Tool frames: {mpc.tool_frames}")
    print("  拖拽绿色控制框 → 机器人实时追踪")
    print("  点击 Coordinate → 拖动任一侧，另一侧自动跟随 (协同)")
    print("  Press Ctrl+C to exit\n")

    step = 0
    left_prev_pos = left_ee_pos.copy()
    right_prev_pos = right_ee_pos.copy()
    left_prev_wxyz = left_ee_wxyz.copy()
    right_prev_wxyz = right_ee_wxyz.copy()

    try:
        while True:
            # ----- 协调模式切换 -----
            if pending_coord_toggle:
                pending_coord_toggle = False
                coordinated = not coordinated

                if coordinated:
                    kin = mpc.compute_kinematics(current_state)
                    tp = kin.tool_poses
                    l_pos = tp[LEFT_TF].position
                    l_quat = tp[LEFT_TF].quaternion
                    r_pos = tp[RIGHT_TF].position
                    r_quat = tp[RIGHT_TF].quaternion

                    # 本地坐标系偏移 (左臂本地)
                    l_pos_np = l_pos.detach().cpu().squeeze().numpy()
                    l_quat_np = l_quat.detach().cpu().squeeze().numpy()
                    r_pos_np = r_pos.detach().cpu().squeeze().numpy()
                    r_quat_np = r_quat.detach().cpu().squeeze().numpy()

                    l_quat_conj_np = _quat_conjugate_np(l_quat_np)
                    coord_delta_pos_local = _quat_rotate_vector_np(
                        l_quat_conj_np, r_pos_np - l_pos_np
                    )
                    coord_delta_quat_local = _quat_multiply_np(
                        l_quat_conj_np, r_quat_np
                    )

                    # 配置 RelativePoseCost
                    from curobo.types import Pose as CuroboPose
                    l_quat_conj = _quat_conjugate(l_quat)
                    rel = CuroboPose(
                        position=_quat_rotate_vector(
                            l_quat_conj, r_pos - l_pos
                        ).clone(),
                        quaternion=quat_multiply(l_quat_conj, r_quat).clone(),
                    )
                    for rollout in mpc.core.get_all_rollout_instances():
                        for mgr in rollout._cost_manager_list:
                            if mgr.has_cost("relative_pose"):
                                cost = mgr.get_cost("relative_pose")
                                cost.target_relative_pose = rel.clone()
                                cost.config.target_relative_pose = rel.clone()
                                cost.enable_cost()

                    coord_btn.color = (255, 165, 0)
                    print(f"[coord] ON — delta={coord_delta_pos_local.tolist()}")
                else:
                    coord_delta_pos_local = None
                    coord_delta_quat_local = None
                    for rollout in mpc.core.get_all_rollout_instances():
                        for mgr in rollout._cost_manager_list:
                            if mgr.has_cost("relative_pose"):
                                mgr.get_cost("relative_pose").disable_cost()
                    coord_btn.color = (128, 128, 128)
                    print("[coord] OFF")

                step = 0
                continue

            # ----- 控制框变化检测 -----
            pose_changed = False
            left_moved = False
            right_moved = False

            if np.any(left_frame.position != left_prev_pos) or \
               np.any(left_frame.wxyz != left_prev_wxyz):
                left_moved = True
            if np.any(right_frame.position != right_prev_pos) or \
               np.any(right_frame.wxyz != right_prev_wxyz):
                right_moved = True

            if not coordinated:
                # 独立模式
                if left_moved:
                    left_prev_pos = left_frame.position.copy()
                    left_prev_wxyz = left_frame.wxyz.copy()
                    pose_changed = True
                if right_moved:
                    right_prev_pos = right_frame.position.copy()
                    right_prev_wxyz = right_frame.wxyz.copy()
                    pose_changed = True

                if pose_changed:
                    dev = current_state.position.device
                    target_link_poses = {
                        LEFT_TF: _numpy_to_pose(
                            left_frame.position, left_frame.wxyz, dev),
                        RIGHT_TF: _numpy_to_pose(
                            right_frame.position, right_frame.wxyz, dev),
                    }
            else:
                # 协同模式: 拖动一方，另一方从本地偏移推算
                if left_moved and not right_moved:
                    left_prev_pos = left_frame.position.copy()
                    left_prev_wxyz = left_frame.wxyz.copy()
                    right_new_pos = left_prev_pos + _quat_rotate_vector_np(
                        left_prev_wxyz, coord_delta_pos_local)
                    right_new_wxyz = _quat_multiply_np(
                        left_prev_wxyz, coord_delta_quat_local)
                    right_frame.position = right_new_pos
                    right_frame.wxyz = right_new_wxyz
                    right_prev_pos = right_new_pos.copy()
                    right_prev_wxyz = right_new_wxyz.copy()
                    pose_changed = True
                elif right_moved and not left_moved:
                    right_prev_pos = right_frame.position.copy()
                    right_prev_wxyz = right_frame.wxyz.copy()
                    delta_quat_conj = _quat_conjugate_np(coord_delta_quat_local)
                    left_new_wxyz = _quat_multiply_np(
                        right_prev_wxyz, delta_quat_conj)
                    left_new_pos = right_prev_pos - _quat_rotate_vector_np(
                        left_new_wxyz, coord_delta_pos_local)
                    left_frame.position = left_new_pos
                    left_frame.wxyz = left_new_wxyz
                    left_prev_pos = left_new_pos.copy()
                    left_prev_wxyz = left_new_wxyz.copy()
                    pose_changed = True
                elif left_moved and right_moved:
                    left_prev_pos = left_frame.position.copy()
                    left_prev_wxyz = left_frame.wxyz.copy()
                    right_prev_pos = right_frame.position.copy()
                    right_prev_wxyz = right_frame.wxyz.copy()
                    pose_changed = True

                if pose_changed:
                    dev = current_state.position.device
                    target_link_poses = {
                        LEFT_TF: _numpy_to_pose(
                            left_prev_pos, left_prev_wxyz, dev),
                        RIGHT_TF: _numpy_to_pose(
                            right_prev_pos, right_prev_wxyz, dev),
                    }

            # ----- 更新 MPC 目标 -----
            if pose_changed:
                mpc.update_goal_tool_poses(
                    GoalToolPose.from_poses(
                        target_link_poses,
                        ordered_tool_frames=mpc.tool_frames,
                        num_goalset=1,
                    ),
                    run_ik=False,
                )
                step = 0

            # ----- Reactive MPC 步进 -----
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
