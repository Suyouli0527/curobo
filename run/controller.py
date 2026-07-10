"""控制层 — MPC 初始化, 协同约束, 主循环."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import GoalToolPose, JointState, Pose
from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg
from curobo._src.geom.quaternion import quat_multiply

from run.utils import (
    numpy_to_pose,
    pose_to_numpy,
    quat_conjugate,
    quat_conjugate_np,
    quat_multiply_np,
    quat_rotate_vector,
    quat_rotate_vector_np,
)


class DualArmMPCController:
    """双臂 MPC 控制器 — 管理 MPC、协同约束和响应式主循环."""

    LEFT_TF = "panda_left_hand_tcp"
    RIGHT_TF = "panda_right_hand_tcp"

    def __init__(
        self,
        robot_file: str = "dual_franka_with_camera.yml",
        scene_dict: Optional[dict] = None,
        optimization_dt: float = 0.025,
        cold_start_iters: int = 300,
        warm_start_iters: int = 200,
    ):
        self.robot_file = robot_file

        # -- MPC 初始化 --
        print(f"Initializing MPC with robot={robot_file}...")
        config = ModelPredictiveControlCfg.create(
            robot=robot_file,
            scene_model=scene_dict,
            self_collision_check=True,
            collision_cache={"cuboid": 50, "mesh": 10},
            optimization_dt=optimization_dt,
            use_cuda_graph=True,
            cold_start_optimization_num_iters=cold_start_iters,
            warm_start_optimization_num_iters=warm_start_iters,
        )
        self.mpc = ModelPredictiveControl(config)

        # 禁用内置 IK solver 的 CUDA graph
        self._disable_ik_cuda_graph()

        # -- 协同约束 --
        self._setup_relative_pose_constraint()

        # -- 初始状态 --
        self.current_state = JointState.from_position(
            self.mpc.default_joint_position.clone().unsqueeze(0),
            joint_names=self.mpc.joint_names,
        )
        self.current_state.velocity = torch.zeros_like(self.current_state.position)
        self.current_state.acceleration = torch.zeros_like(self.current_state.position)

        print("Setting up MPC...")
        self.mpc.setup(self.current_state)
        print("  MPC setup complete")

        # -- 协同状态 --
        self.coordinated = False
        self.coord_delta_pos_local: Optional[np.ndarray] = None
        self.coord_delta_quat_local: Optional[np.ndarray] = None

    # ================================================================
    #  内部方法
    # ================================================================

    def _disable_ik_cuda_graph(self) -> None:
        try:
            for key in list(self.mpc.ik_solver.__dict__.keys()):
                if "seed" in key and hasattr(
                    getattr(self.mpc.ik_solver, key), "config"
                ):
                    getattr(self.mpc.ik_solver, key).config.use_cuda_graph = False
        except Exception:
            pass

    def _setup_relative_pose_constraint(self) -> None:
        """注入 soft (梯度引导) + hard (二元约束) 两条路径."""
        rel_pose_cfg = RelativePoseCostCfg(
            weight=torch.tensor([50000.0, 5000.0]),
            primary_tool_frame=self.LEFT_TF,
            secondary_tool_frame=self.RIGHT_TF,
            position_tolerance=0.01,
            orientation_tolerance=0.05,
        )
        rel_pose_hard_cfg = RelativePoseCostCfg(
            weight=torch.tensor([500000.0, 50000.0]),
            primary_tool_frame=self.LEFT_TF,
            secondary_tool_frame=self.RIGHT_TF,
            position_tolerance=0.01,
            orientation_tolerance=0.05,
            convert_to_binary=True,
        )
        core = self.mpc.config.core_cfg
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
        for rollout in self.mpc.core.get_all_rollout_instances():
            for mgr in rollout._cost_manager_list:
                if mgr.has_cost("relative_pose"):
                    mgr.get_cost("relative_pose").disable_cost()

    def _enable_relative_pose(self) -> None:
        """捕获当前相对位姿并启用约束."""
        kin = self.mpc.compute_kinematics(self.current_state)
        tp = kin.tool_poses
        l_pos = tp[self.LEFT_TF].position
        l_quat = tp[self.LEFT_TF].quaternion
        r_pos = tp[self.RIGHT_TF].position
        r_quat = tp[self.RIGHT_TF].quaternion

        # numpy 偏移 (本地坐标系)
        l_pos_np = l_pos.detach().cpu().squeeze().numpy()
        l_quat_np = l_quat.detach().cpu().squeeze().numpy()
        r_pos_np = r_pos.detach().cpu().squeeze().numpy()
        r_quat_np = r_quat.detach().cpu().squeeze().numpy()

        l_quat_conj_np = quat_conjugate_np(l_quat_np)
        self.coord_delta_pos_local = quat_rotate_vector_np(
            l_quat_conj_np, r_pos_np - l_pos_np
        )
        self.coord_delta_quat_local = quat_multiply_np(
            l_quat_conj_np, r_quat_np
        )

        # torch 约束目标
        from curobo.types import Pose as CuroboPose
        l_quat_conj = quat_conjugate(l_quat)
        rel = CuroboPose(
            position=quat_rotate_vector(l_quat_conj, r_pos - l_pos).clone(),
            quaternion=quat_multiply(l_quat_conj, r_quat).clone(),
        )
        for rollout in self.mpc.core.get_all_rollout_instances():
            for mgr in rollout._cost_manager_list:
                if mgr.has_cost("relative_pose"):
                    cost = mgr.get_cost("relative_pose")
                    cost.target_relative_pose = rel.clone()
                    cost.config.target_relative_pose = rel.clone()
                    cost.enable_cost()
        print(f"[coord] ON — delta={self.coord_delta_pos_local.tolist()}")

    def _disable_relative_pose(self) -> None:
        self.coord_delta_pos_local = None
        self.coord_delta_quat_local = None
        for rollout in self.mpc.core.get_all_rollout_instances():
            for mgr in rollout._cost_manager_list:
                if mgr.has_cost("relative_pose"):
                    mgr.get_cost("relative_pose").disable_cost()
        print("[coord] OFF")

    # ================================================================
    #  公共接口
    # ================================================================

    def toggle_coordination(self) -> None:
        """切换协调模式."""
        self.coordinated = not self.coordinated
        if self.coordinated:
            self._enable_relative_pose()
        else:
            self._disable_relative_pose()

    def compute_ee_poses(self):
        """返回当前末端位姿 (numpy)."""
        kin = self.mpc.compute_kinematics(self.current_state)
        tp = kin.tool_poses
        return (
            pose_to_numpy(tp[self.LEFT_TF].position, tp[self.LEFT_TF].quaternion),
            pose_to_numpy(tp[self.RIGHT_TF].position, tp[self.RIGHT_TF].quaternion),
        )

    def set_initial_goal(self, left_pose, right_pose) -> None:
        """设置初始目标位姿 (不跑 IK)."""
        left_pos, left_wxyz = left_pose
        right_pos, right_wxyz = right_pose
        dev = self.current_state.position.device
        self.mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(
                {
                    self.LEFT_TF: numpy_to_pose(left_pos, left_wxyz, dev),
                    self.RIGHT_TF: numpy_to_pose(right_pos, right_wxyz, dev),
                },
                ordered_tool_frames=self.mpc.tool_frames,
                num_goalset=1,
            ),
            run_ik=False,
        )

    def update_goal(self, left_pose, right_pose) -> None:
        """更新 Cartesian 目标."""
        left_pos, left_wxyz = left_pose
        right_pos, right_wxyz = right_pose
        dev = self.current_state.position.device
        self.mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(
                {
                    self.LEFT_TF: numpy_to_pose(left_pos, left_wxyz, dev),
                    self.RIGHT_TF: numpy_to_pose(right_pos, right_wxyz, dev),
                },
                ordered_tool_frames=self.mpc.tool_frames,
                num_goalset=1,
            ),
            run_ik=False,
        )

    def step(self) -> bool:
        """执行一步 MPC 优化, 返回是否有有效动作."""
        result = self.mpc.optimize_action_sequence(self.current_state)
        if (
            result is not None
            and result.action_sequence is not None
            and result.action_sequence.position.shape[1] > 0
        ):
            next_position = result.action_sequence.position[:, -1, :]
            self.current_state = JointState.from_position(
                next_position.clone(),
                joint_names=self.mpc.joint_names,
            )
            self.current_state.velocity = result.action_sequence.velocity[:, -1, :]
            self.current_state.acceleration = result.action_sequence.acceleration[:, -1, :]
            return True
        return False

    @property
    def joint_names(self):
        return self.mpc.joint_names

    @property
    def tool_frames(self):
        return self.mpc.tool_frames

    def destroy(self):
        self.mpc.destroy()

    # ================================================================
    #  协同框同步
    # ================================================================

    def sync_frame_left_to_right(
        self, left_pos: np.ndarray, left_wxyz: np.ndarray
    ) -> tuple:
        """左框 → 右框 (本地偏移推算)."""
        right_pos = left_pos + quat_rotate_vector_np(
            left_wxyz, self.coord_delta_pos_local
        )
        right_wxyz = quat_multiply_np(left_wxyz, self.coord_delta_quat_local)
        return right_pos, right_wxyz

    def sync_frame_right_to_left(
        self, right_pos: np.ndarray, right_wxyz: np.ndarray
    ) -> tuple:
        """右框 → 左框 (反算)."""
        delta_quat_conj = quat_conjugate_np(self.coord_delta_quat_local)
        left_wxyz = quat_multiply_np(right_wxyz, delta_quat_conj)
        left_pos = right_pos - quat_rotate_vector_np(
            left_wxyz, self.coord_delta_pos_local
        )
        return left_pos, left_wxyz
