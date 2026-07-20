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
        cold_start_iters: int = 100,
        warm_start_iters: int = 100,
        master_slave: bool = False,
        split_spheres: bool = False,
    ):
        self.robot_file = robot_file
        self.master_slave = master_slave
        self.split_spheres = split_spheres

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
        # 必须在 MPC 构建前注入 relative_pose，否则 cost manager 不会创建该 cost
        self._inject_relative_pose_config(config)
        self.mpc = ModelPredictiveControl(config)

        # 禁用内置 IK solver 的 CUDA graph
        self._disable_ik_cuda_graph()

        # -- 协同约束（运行时开关）--
        self._init_relative_pose_control()

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

        # -- 抓取状态 --
        self._attached_name: Optional[str] = None

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

    def _inject_relative_pose_config(self, config) -> None:
        """在 MPC 构建前，把 relative_pose cost 注入 config."""
        self._rel_pose_cfg = RelativePoseCostCfg(
            weight=torch.tensor([7500.0, 750.0]),
            primary_tool_frame=self.LEFT_TF,
            secondary_tool_frame=self.RIGHT_TF,
            position_tolerance=0.01,
            orientation_tolerance=0.05,
        )
        rel_pose_hard_cfg = RelativePoseCostCfg(
            weight=torch.tensor([75000.0, 7500.0]),
            primary_tool_frame=self.LEFT_TF,
            secondary_tool_frame=self.RIGHT_TF,
            position_tolerance=0.01,
            orientation_tolerance=0.05,
            convert_to_binary=True,
        )
        core = config.core_cfg
        for rc in core.optimizer_rollout_configs:
            if rc.cost_cfg is not None:
                rc.cost_cfg.relative_pose_cfg = self._rel_pose_cfg
            if rc.constraint_cfg is not None:
                rc.constraint_cfg.relative_pose_cfg = rel_pose_hard_cfg
        mr = core.metrics_rollout_config
        if mr is not None:
            if mr.cost_cfg is not None:
                mr.cost_cfg.relative_pose_cfg = self._rel_pose_cfg
            if mr.constraint_cfg is not None:
                mr.constraint_cfg.relative_pose_cfg = rel_pose_hard_cfg

    def _init_relative_pose_control(self) -> None:
        """MPC 构建后，初始禁用 relative_pose."""
        for rollout in self.mpc.core.get_all_rollout_instances():
            for mgr in rollout._cost_manager_list:
                if mgr.has_cost("relative_pose"):
                    mgr.get_cost("relative_pose").disable_cost()

    def _enable_relative_pose(
        self, left_frame=None, right_frame=None,
    ) -> None:
        """捕获当前相对位姿并启用约束.

        left_frame / right_frame: (pos, wxyz) numpy tuple.
        用于框同步的 delta 从目标框取, 不是 FK; 约束目标仍用 FK.
        """
        kin = self.mpc.compute_kinematics(self.current_state)
        tp = kin.tool_poses
        l_fk_pos = tp[self.LEFT_TF].position
        l_fk_quat = tp[self.LEFT_TF].quaternion
        r_fk_pos = tp[self.RIGHT_TF].position
        r_fk_quat = tp[self.RIGHT_TF].quaternion

        # 框同步 delta: 从目标框取 (如果提供), 否则用 FK 兜底
        if left_frame is not None and right_frame is not None:
            l_np, lq_np = left_frame
            r_np, rq_np = right_frame
        else:
            l_np = l_fk_pos.detach().cpu().squeeze().numpy()
            lq_np = l_fk_quat.detach().cpu().squeeze().numpy()
            r_np = r_fk_pos.detach().cpu().squeeze().numpy()
            rq_np = r_fk_quat.detach().cpu().squeeze().numpy()

        l_quat_conj_np = quat_conjugate_np(lq_np)
        self.coord_delta_pos_local = quat_rotate_vector_np(
            l_quat_conj_np, r_np - l_np
        )
        self.coord_delta_quat_local = quat_multiply_np(
            l_quat_conj_np, rq_np
        )

        # torch 约束目标: 始终用 FK (真机相对位姿)
        from curobo.types import Pose as CuroboPose
        l_quat_conj = quat_conjugate(l_fk_quat)
        rel = CuroboPose(
            position=quat_rotate_vector(l_quat_conj, r_fk_pos - l_fk_pos).clone(),
            quaternion=quat_multiply(l_quat_conj, r_fk_quat).clone(),
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

    def toggle_coordination(self, left_frame=None, right_frame=None) -> None:
        if self._attached_name is not None: return
        self.coordinated = not self.coordinated
        if self.coordinated: self._enable_relative_pose(left_frame, right_frame)
        else: self._disable_relative_pose()

    def compute_ee_poses(self):
        """返回当前末端位姿 (numpy)."""
        kin = self.mpc.compute_kinematics(self.current_state)
        tp = kin.tool_poses
        return (
            pose_to_numpy(tp[self.LEFT_TF].position, tp[self.LEFT_TF].quaternion),
            pose_to_numpy(tp[self.RIGHT_TF].position, tp[self.RIGHT_TF].quaternion),
        )

    def compute_rel_pose_error(self):
        """返回当前相对位姿与目标之间的偏差 [pos_err_mm, rot_err_deg]."""
        kin = self.mpc.compute_kinematics(self.current_state)
        tp = kin.tool_poses
        lq = tp[self.LEFT_TF].quaternion
        lp = tp[self.LEFT_TF].position
        rq = tp[self.RIGHT_TF].quaternion
        rp = tp[self.RIGHT_TF].position
        # 当前相对位姿
        lq_conj = quat_conjugate(lq)
        cur_rel_pos = quat_rotate_vector(lq_conj, rp - lp)
        cur_rel_quat = quat_multiply(lq_conj, rq)
        # 目标相对位姿 (存于第一个 rollout 的 relative_pose cost 中)
        target_rel_pos = None
        target_rel_quat = None
        for rollout in self.mpc.core.get_all_rollout_instances():
            for mgr in rollout._cost_manager_list:
                if mgr.has_cost("relative_pose"):
                    cost = mgr.get_cost("relative_pose")
                    if cost.enabled:
                        target_rel_pos = cost.target_relative_pose.position
                        target_rel_quat = cost.target_relative_pose.quaternion
                        break
        if target_rel_pos is None:
            return 0.0, 0.0
        pos_err = torch.norm(cur_rel_pos - target_rel_pos).item() * 1000.0  # mm
        # orientation error: angle of quaternion difference
        dq = quat_multiply(quat_conjugate(cur_rel_quat), target_rel_quat)
        dq = dq / (torch.norm(dq) + 1e-8)
        rot_err = 2.0 * torch.acos(torch.clamp(torch.abs(dq[..., 0]), 0.0, 1.0)).item() * 180.0 / 3.14159
        return pos_err, rot_err

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
        """更新 Cartesian 目标 (与 reative_control example 保持一致)."""
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

    def step(self) -> dict:
        """执行一步 MPC 优化, 返回 {success, solve_time, min_scene_dist, min_self_dist}.

        min_scene_dist / min_self_dist 是当前状态下的最小碰撞距离 (米),
        正数=安全距离, 负数=穿透深度.
        """
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

            # --- 提取实际碰撞距离 (米, 正=安全, 负=穿透) ---
            min_scene = self._get_min_scene_distance()
            min_self = self._get_min_self_distance()
            if not hasattr(self, "_dbg_cmp"):
                self._dbg_cmp = True
                mpc_v = self._get_mpc_scene_collision_value()
                print(f"[debug cmp] AABB_min_dist={min_scene*1000:.1f}mm  "
                      f"MPC_constraint_max={mpc_v:.2f}  "
                      f"self_col={min_self*1000:.1f}mm")

            pos_err = result.position_error
            if pos_err is not None:
                pos_err = float(pos_err.max().item())
            # 提取 objective value
            obj_val = 0.0
            try:
                m = self.mpc.trajectory_execution_manager.get_current_metrics()
                s = m.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=True)
                if s is not None:
                    obj_val = float(s.detach().cpu().item())
            except Exception:
                pass

            return {
                "success": True,
                "solve_time": result.solve_time,
                "min_scene_dist": min_scene,
                "min_self_dist": min_self,
                "pos_err": pos_err,
                "obj_val": obj_val,
            }
        return {"success": False, "solve_time": 0.0, "min_scene_dist": 0.0, "min_self_dist": 0.0,
                "pos_err": None, "obj_val": 0.0}

    def _get_min_scene_distance(self) -> float:
        """计算机器人球体到场景 cuboid 的最短距离 (米), 正=间隙, 负=穿透.

        使用 Torch 向量化 AABB 距离计算 (忽略旋转, 对大多数轴对齐场景足够).
        """
        sc = self.mpc.scene_collision_checker
        if sc is None:
            return 1.0
        cdata = sc.data.cuboids  # SceneData (GPU), not SceneCfg
        if cdata is None or cdata.count[0].item() == 0:
            return 1.0

        kin = self.mpc.compute_kinematics(self.current_state)
        spheres = kin.robot_spheres.squeeze()  # [N, 4]
        active = spheres[:, 3] > 1e-6
        if not active.any():
            return 1.0

        sc_m = spheres[active, :3]  # [M, 3]
        sr = spheres[active, 3]    # [M]

        n_cub = int(cdata.count[0].item())
        # 只取启用的 cuboid
        enabled = cdata.enable[0, :n_cub].bool()
        if not enabled.any():
            return 1.0
        half = cdata.dims[0, :n_cub, :3][enabled] / 2.0       # [E, 3]
        c_pos = -cdata.inv_pose[0, :n_cub, :3][enabled]        # [E, 3]
        c_min = c_pos - half                                     # [E, 3]
        c_max = c_pos + half                                     # [E, 3]

        # [M, 1, 3] vs [1, K, 3] -> closest point on each cuboid AABB
        closest = torch.clamp(sc_m.unsqueeze(1), c_min.unsqueeze(0), c_max.unsqueeze(0))
        dist = torch.norm(sc_m.unsqueeze(1) - closest, dim=2)  # [M, K]
        signed = dist - sr.unsqueeze(1)                        # [M, K]

        # 同时查询 MPC 优化器自己的碰撞值 (对比)
        mpc_col = self._get_mpc_scene_collision_value()

        return float(signed.min().item())

    def _get_mpc_scene_collision_value(self) -> float:
        """读取 MPC 优化器最后一次计算出的场景碰撞约束最大值 (非 binary)."""
        try:
            for rollout in self.mpc.core.get_all_rollout_instances():
                for mgr in rollout._cost_manager_list:
                    if not mgr.has_cost("scene_collision"):
                        continue
                    cost = mgr.get_cost("scene_collision")
                    if not cost.enabled:
                        continue
                    if getattr(cost.config, "convert_to_binary", False):
                        continue
                    if hasattr(cost, "_collision_buffer") and cost._collision_buffer is not None:
                        d = cost._collision_buffer.distance.detach()
                        if d.numel() > 0:
                            return float(d.max().item())
        except Exception:
            pass
        return 0.0

    def _get_min_self_distance(self) -> float:
        """读取自碰撞穿透深度 (米). 0=安全, <0=穿透. 复用已有 _out_distance."""
        try:
            for rollout in self.mpc.core.get_all_rollout_instances():
                for mgr in rollout._cost_manager_list:
                    if not mgr.has_cost("self_collision"):
                        continue
                    cost = mgr.get_cost("self_collision")
                    if not cost.enabled:
                        continue
                    if getattr(cost.config, "convert_to_binary", False):
                        continue
                    if not hasattr(cost, "_out_distance") or cost._out_distance.numel() == 0:
                        continue
                    d = cost._out_distance.detach()
                    w = cost._weight.detach().item()
                    return float(-d.max().item() / w)
        except Exception:
            pass
        return 0.0

    @property
    def joint_names(self):
        return self.mpc.joint_names

    @property
    def tool_frames(self):
        return self.mpc.tool_frames

    def destroy(self):
        self.mpc.destroy()

    # ================================================================
    #  抓取 / 释放
    # ================================================================

    def grasp_obstacle(self, obstacle_name: str, left_frame=None, right_frame=None) -> bool:
        am = self.mpc.core.attachment_manager
        sc = self.mpc.core.scene_collision_checker

        sm = sc.scene_model
        if isinstance(sm, list): sm = sm[0]
        obs = sm.get_obstacle(obstacle_name)
        if obs is None: return False

        # 1. 隐藏原障碍物
        sc.enable_obstacle(obstacle_name, enable=False, env_idx=0)

        # 2. 球体拟合
        spheres = am.fit_spheres([obs], num_spheres=16, surface_radius=0.005)
        dev, dt = spheres.device, spheres.dtype
        print(f"[grasp] sphere radii: min={spheres[:,3].min().item()*1000:.1f}mm  "
              f"max={spheres[:,3].max().item()*1000:.1f}mm  "
              f"mean={spheres[:,3].mean().item()*1000:.1f}mm")

        # 3. 计算偏移
        kin = self.mpc.compute_kinematics(self.current_state)
        lp = kin.tool_poses[self.LEFT_TF].position.squeeze()
        rp = kin.tool_poses[self.RIGHT_TF].position.squeeze()
        obs_center = torch.tensor(obs.pose[:3], device=dev, dtype=dt)
        offset = Pose(
            position=((lp + rp) / 2.0 - obs_center).view(1, 3),
            quaternion=torch.tensor([[1.0, 0, 0, 0]], device=dev, dtype=dt),
        )

        # 4. 启动协同 (用目标框位姿算 delta)
        if not self.coordinated:
            self._enable_relative_pose(left_frame, right_frame)
            self.coordinated = True

        # 5. 附着球体
        if self.split_spheres:
            sorted_idx = torch.argsort(spheres[:, 1])
            n = spheres.shape[0]
            am.update(spheres[sorted_idx[:n // 2]], self.current_state,
                      link_name="attached_object_left",
                      world_objects_pose_offset=offset, ee_link=self.LEFT_TF)
            am.update(spheres[sorted_idx[n // 2:]], self.current_state,
                      link_name="attached_object_right",
                      world_objects_pose_offset=offset, ee_link=self.RIGHT_TF)
            print(f"[grasp] {n} spheres split L:{n//2} R:{n - n//2}")
        else:
            am.update(spheres, self.current_state,
                      link_name="attached_object_left",
                      world_objects_pose_offset=offset, ee_link=self.LEFT_TF)
            if not self.master_slave:
                am.update(spheres, self.current_state,
                          link_name="attached_object_right",
                          world_objects_pose_offset=offset, ee_link=self.RIGHT_TF)
            mode = "master-slave" if self.master_slave else "dual"
            print(f"[grasp] {spheres.shape[0]} spheres ({mode})")

        # 6. 轻量冷启动 + 热启动
        self._attached_name = obstacle_name
        self.mpc.reset_cuda_graph()
        self.mpc._mpc_warm_start_available = True

        # 诊断：打印附着后的碰撞距离
        min_d = self._get_min_scene_distance()
        print(f"[grasp] '{obstacle_name}' attached ({spheres.shape[0]} spheres)  "
              f"min_dist={min_d*1000:.1f}mm")
        return True

    def release_obstacle(self) -> bool:
        if self._attached_name is None: return False
        am = self.mpc.core.attachment_manager
        sc = self.mpc.core.scene_collision_checker
        am.detach(link_name="attached_object_left")
        am.detach(link_name="attached_object_right")
        sc.enable_obstacle(self._attached_name, enable=True, env_idx=0)
        name = self._attached_name
        self._attached_name = None
        self.mpc.reset_cuda_graph()
        self.mpc._mpc_warm_start_available = False  # 下次 step 自动 cold_start
        print(f"[release] '{name}' detached")
        return True

    @property
    def attached_name(self) -> Optional[str]:
        return self._attached_name

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
