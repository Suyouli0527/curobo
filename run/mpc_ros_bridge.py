"""MPC ROS2 桥接节点 — 将 cuRobo MPC 接入 MuJoCo 仿真.

支持双臂协同约束: MPC 优化器内的 RelativePoseCost 自动维持左右手末端之间的
固定相对位姿, 因此只需要指定一只手的目标, 另一只手会自动跟随.

参考 baseline1 的 LowROSModule 模式: 独立的 ROS2 Node, 订阅关节状态,
运行 MPC 优化, 发布关节位置命令.

用法:
    source ~/csc_curobo/.venv/bin/activate
    source /opt/ros/humble/setup.bash
    python ~/csc_curobo/run/mpc_ros_bridge.py
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

# 添加 csc_curobo 到 Python path
_CUR_DIR = Path(__file__).resolve().parent
if str(_CUR_DIR) not in sys.path:
    sys.path.insert(0, str(_CUR_DIR))

from controller import DualArmMPCController


# ==========================================================================
# 关节名称映射: MuJoCo ROS2 ↔ cuRobo
# ==========================================================================

# cuRobo 双臂关节顺序 (dual_franka_with_camera.yml):
#   [panda_right_j1..j7, panda_right_f1..f2, panda_left_j1..j7, panda_left_f1..f2]
# 夹爪 (f1, f2) 锁定, 活动关节 = 索引 0-6 (右手), 9-15 (左手)

CUROBO_TO_ROS_JOINT = {
    "panda_right_joint1": "mj_right_joint1",
    "panda_right_joint2": "mj_right_joint2",
    "panda_right_joint3": "mj_right_joint3",
    "panda_right_joint4": "mj_right_joint4",
    "panda_right_joint5": "mj_right_joint5",
    "panda_right_joint6": "mj_right_joint6",
    "panda_right_joint7": "mj_right_joint7",
    "panda_left_joint1": "mj_left_joint1",
    "panda_left_joint2": "mj_left_joint2",
    "panda_left_joint3": "mj_left_joint3",
    "panda_left_joint4": "mj_left_joint4",
    "panda_left_joint5": "mj_left_joint5",
    "panda_left_joint6": "mj_left_joint6",
    "panda_left_joint7": "mj_left_joint7",
}

ROS_TO_CUROBO_JOINT = {v: k for k, v in CUROBO_TO_ROS_JOINT.items()}

CUROBO_JOINT_NAMES_ACTIVE = [
    "panda_right_joint1", "panda_right_joint2", "panda_right_joint3",
    "panda_right_joint4", "panda_right_joint5", "panda_right_joint6",
    "panda_right_joint7",
    "panda_left_joint1", "panda_left_joint2", "panda_left_joint3",
    "panda_left_joint4", "panda_left_joint5", "panda_left_joint6",
    "panda_left_joint7",
]

ROS_JOINT_NAMES_ACTIVE = [CUROBO_TO_ROS_JOINT[n] for n in CUROBO_JOINT_NAMES_ACTIVE]


# ==========================================================================
# MPC ROS2 桥接节点 (支持协同约束)
# ==========================================================================

class MPCRosBridge(Node):
    """桥接 MuJoCo 仿真与 cuRobo MPC, 支持双臂协同约束.

    协同约束原理:
      - MPC 优化器内注入 RelativePoseCost, 自动维持左右手末端之间的固定相对位姿
      - 通过 ROS2 服务 /mpc/toggle_coordination 开关协同
      - 协同开启时: 只需给一只手目标位姿, 另一只自动跟随
      - 协同关闭时: 双臂各自独立跟踪各自目标

    Topics:
      /joint_states  (sub)  — 仿真关节状态
      /mpc_joint_commands (pub) — MPC 输出的关节位置命令
      /mpc/target_pose_left  (sub) — 左手目标位姿 (geometry_msgs/Pose)
      /mpc/target_pose_right (sub) — 右手目标位姿

    Services:
      /mpc/toggle_coordination (std_srvs/SetBool) — 开关协同约束
    """

    LEFT_TF = "panda_left_hand_tcp"
    RIGHT_TF = "panda_right_hand_tcp"

    def __init__(
        self,
        robot_file: str = "dual_franka_with_camera.yml",
        optimization_dt: float = 0.025,
        cold_start_iters: int = 100,
        warm_start_iters: int = 200,
    ):
        super().__init__("mpc_ros_bridge")

        self.mpc_dt = optimization_dt

        # -- 线程安全 --
        self._lock = threading.Lock()
        self._latest_joint_state: Optional[JointState] = None
        self._first_state_received = False

        # -- 目标缓存 --
        self._target_left: Optional[Pose] = None   # geometry_msgs/Pose
        self._target_right: Optional[Pose] = None

        # -- MPC 控制器 (完全复用 DualArmMPCController) --
        print(f"[MPC Bridge] Initializing MPC with robot={robot_file}...")
        self.mpc_controller = DualArmMPCController(
            robot_file=robot_file,
            optimization_dt=optimization_dt,
            cold_start_iters=cold_start_iters,
            warm_start_iters=warm_start_iters,
        )
        # 初始状态: 禁用协同 (双臂独立 hold position)
        # DualArmMPCController.__init__ 已经调用了 _init_relative_pose_control(),
        # 协同约束已存在但被禁用。下面再确保一次。
        self.mpc_controller.coordinated = False
        print(f"[MPC Bridge] MPC initialized, coordination: OFF")

        # -- ROS2 通信 --
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 1)
        self.cmd_pub = self.create_publisher(
            JointState, "/mpc_joint_commands", 1)

        # 目标位姿订阅 (左右各一, geometry_msgs/Pose)
        self.target_left_sub = self.create_subscription(
            Pose, "/mpc/target_pose_left", self._target_left_cb, 1)
        self.target_right_sub = self.create_subscription(
            Pose, "/mpc/target_pose_right", self._target_right_cb, 1)

        # 协同开关服务
        self.coord_srv = self.create_service(
            SetBool, "/mpc/toggle_coordination", self._toggle_coordination_cb)

        # -- MPC Timer --
        self._timer = self.create_timer(self.mpc_dt, self._mpc_step)

        # -- 统计 --
        self._step_count = 0
        self._total_solve_time = 0.0
        self._last_report = time.time()

        print(f"[MPC Bridge] Ready. Topics: cmd→/mpc_joint_commands, "
              f"target→/mpc/target_pose_{'{left,right}'}, "
              f"srv→/mpc/toggle_coordination")

    # ======================================================================
    # ROS2 Callbacks
    # ======================================================================

    def _joint_state_cb(self, msg: JointState):
        with self._lock:
            self._latest_joint_state = msg
            if not self._first_state_received:
                self._first_state_received = True
                self._on_first_state(msg)

    def _target_left_cb(self, msg: Pose):
        """接收左手目标位姿 (geometry_msgs/Pose)."""
        self._target_left = msg
        self.get_logger().info(
            f"Left target: pos=[{msg.position.x:.3f}, {msg.position.y:.3f}, "
            f"{msg.position.z:.3f}]",
            throttle_duration_sec=0.5)

    def _target_right_cb(self, msg: Pose):
        """接收右手目标位姿."""
        self._target_right = msg
        self.get_logger().info(
            f"Right target: pos=[{msg.position.x:.3f}, {msg.position.y:.3f}, "
            f"{msg.position.z:.3f}]",
            throttle_duration_sec=0.5)

    def _toggle_coordination_cb(self, request, response):
        """ROS2 服务: 开关双臂协同约束."""
        enable = request.data
        try:
            if enable and not self.mpc_controller.coordinated:
                self.mpc_controller.toggle_coordination()
                response.success = True
                response.message = "Coordination ON — relative pose constraint enabled"
            elif not enable and self.mpc_controller.coordinated:
                self.mpc_controller.toggle_coordination()
                response.success = True
                response.message = "Coordination OFF — independent arm control"
            else:
                response.success = True
                response.message = f"Already {'ON' if enable else 'OFF'}"
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response

    def _on_first_state(self, msg: JointState):
        """首次收到关节状态: 将 MPC 重置到仿真当前状态."""
        self.get_logger().info(f"First joint state received: {len(msg.name)} joints")
        try:
            current_state = self._ros_to_curobo_state(msg)
            # 初始目标 = 保持当前末端位姿
            self._set_hold_goal(current_state)
            self.mpc_controller.mpc.reset_robot(current_state)
            self.mpc_controller.current_state = current_state
            self.mpc_controller.mpc._mpc_warm_start_available = False
            self.get_logger().info("MPC reset to current simulation state")
        except Exception as e:
            self.get_logger().error(f"Failed to reset MPC: {e}")

    # ======================================================================
    # 状态转换
    # ======================================================================

    def _ros_to_curobo_state(self, msg: JointState):
        """ROS2 JointState → cuRobo JointState (18维, 含锁定夹爪)."""
        ros_positions = dict(zip(msg.name, msg.position))

        n_full = len(self.mpc_controller.joint_names)  # 18
        positions = torch.zeros(1, n_full, device="cuda")

        for curobo_name, ros_name in CUROBO_TO_ROS_JOINT.items():
            if ros_name in ros_positions:
                idx = self.mpc_controller.joint_names.index(curobo_name)
                positions[0, idx] = float(ros_positions[ros_name])

        # 锁定夹爪
        for lock_name, lock_val in self.mpc_controller.mpc.kinematics.lock_joints.items():
            if lock_name in self.mpc_controller.joint_names:
                idx = self.mpc_controller.joint_names.index(lock_name)
                positions[0, idx] = lock_val

        from curobo.types import JointState as CuroboJointState
        return CuroboJointState.from_position(
            positions, joint_names=self.mpc_controller.joint_names)

    def _curobo_to_ros_position(self) -> List[float]:
        """cuRobo current_state → ROS2 14维位置列表."""
        state = self.mpc_controller.current_state
        positions = state.position.squeeze(0)  # [18]
        return [
            float(positions[self.mpc_controller.joint_names.index(n)].cpu())
            for n in CUROBO_JOINT_NAMES_ACTIVE
        ]

    # ======================================================================
    # 目标管理 (核心: 协同感知)
    # ======================================================================

    def _pose_msg_to_numpy(self, pose_msg: Pose):
        """geometry_msgs/Pose → (pos_xyz, quat_wxyz) numpy."""
        pos = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
        quat = np.array([pose_msg.orientation.w, pose_msg.orientation.x,
                         pose_msg.orientation.y, pose_msg.orientation.z])
        return pos, quat

    def _set_hold_goal(self, current_state):
        """设置目标为保持当前末端位姿."""
        kin = self.mpc_controller.mpc.compute_kinematics(current_state)
        tool_frames = self.mpc_controller.tool_frames
        goal_poses = kin.tool_poses.as_goal(tool_frames)
        from curobo.types import GoalToolPose
        self.mpc_controller.mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(
                goal_poses.to_dict(),
                ordered_tool_frames=tool_frames,
                num_goalset=1,
            ),
            run_ik=False,
        )

    def _apply_targets(self):
        """将收到的目标位姿应用到 MPC, 支持协同模式下的自动同步."""
        left_target = self._target_left
        right_target = self._target_right

        if left_target is None and right_target is None:
            return  # 没有新目标, 保持当前

        coordinated = self.mpc_controller.coordinated

        if coordinated:
            # -- 协同模式: 一只手指定, 另一只自动跟随 --
            if left_target is not None:
                # 左手指定 → 同步右手
                l_pos, l_quat = self._pose_msg_to_numpy(left_target)
                r_pos, r_quat = self.mpc_controller.sync_frame_left_to_right(l_pos, l_quat)
                self._update_dual_goal((l_pos, l_quat), (r_pos, r_quat))
            elif right_target is not None:
                # 右手指定 → 同步左手
                r_pos, r_quat = self._pose_msg_to_numpy(right_target)
                l_pos, l_quat = self.mpc_controller.sync_frame_right_to_left(r_pos, r_quat)
                self._update_dual_goal((l_pos, l_quat), (r_pos, r_quat))
        else:
            # -- 独立模式: 各自跟踪各自目标 --
            l_pose = (self._pose_msg_to_numpy(left_target)
                      if left_target is not None else None)
            r_pose = (self._pose_msg_to_numpy(right_target)
                      if right_target is not None else None)

            if l_pose is not None and r_pose is not None:
                self._update_dual_goal(l_pose, r_pose)
            elif l_pose is not None:
                self._update_single_goal(self.LEFT_TF, l_pose)
            elif r_pose is not None:
                self._update_single_goal(self.RIGHT_TF, r_pose)

        # 清除已处理的目标
        self._target_left = None
        self._target_right = None

    def _update_dual_goal(self, left_pose, right_pose):
        """更新双臂 Cartesian 目标 (不跑 IK, MPC 自己处理)."""
        dev = self.mpc_controller.current_state.position.device
        from curobo.types import GoalToolPose
        from run.utils import numpy_to_pose
        self.mpc_controller.mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(
                {
                    self.LEFT_TF: numpy_to_pose(left_pose[0], left_pose[1], dev),
                    self.RIGHT_TF: numpy_to_pose(right_pose[0], right_pose[1], dev),
                },
                ordered_tool_frames=self.mpc_controller.tool_frames,
                num_goalset=1,
            ),
            run_ik=False,
        )

    def _update_single_goal(self, tool_frame, pose):
        """更新单臂 Cartesian 目标."""
        pos, quat = pose
        dev = self.mpc_controller.current_state.position.device
        from curobo.types import GoalToolPose
        from run.utils import numpy_to_pose
        self.mpc_controller.mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(
                {tool_frame: numpy_to_pose(pos, quat, dev)},
                ordered_tool_frames=self.mpc_controller.tool_frames,
                num_goalset=1,
            ),
            run_ik=False,
        )

    # ======================================================================
    # MPC 控制循环
    # ======================================================================

    def _mpc_step(self):
        """主循环: 读状态 → 应用目标 → MPC step → 发布命令."""
        with self._lock:
            if self._latest_joint_state is None:
                return
            msg = self._latest_joint_state

        try:
            # 1. ROS2 → cuRobo
            curobo_state = self._ros_to_curobo_state(msg)
            curobo_state.velocity = torch.zeros(
                1, len(self.mpc_controller.joint_names), device="cuda")
            curobo_state.acceleration = torch.zeros_like(curobo_state.velocity)
            self.mpc_controller.current_state = curobo_state

            # 2. 应用新目标 (如果有)
            self._apply_targets()

            # 3. MPC step
            result = self.mpc_controller.step()

            # 4. cuRobo → ROS2
            ros_positions = self._curobo_to_ros_position()

            # 5. 发布命令
            cmd_msg = JointState()
            cmd_msg.header.stamp = self.get_clock().now().to_msg()
            cmd_msg.name = ROS_JOINT_NAMES_ACTIVE
            cmd_msg.position = ros_positions
            self.cmd_pub.publish(cmd_msg)

            # 6. 统计
            self._step_count += 1
            solve_time = result.get("solve_time", 0.0)
            self._total_solve_time += solve_time

            now = time.time()
            if now - self._last_report > 5.0:
                avg = self._total_solve_time / max(self._step_count, 1) * 1000
                pos_err = (result.get("pos_err", 0.0) or 0.0) * 1000
                coord = "ON" if self.mpc_controller.coordinated else "OFF"
                self.get_logger().info(
                    f"Step {self._step_count}: solve={avg:.1f}ms "
                    f"pos_err={pos_err:.1f}mm coord={coord}")
                self._last_report = now

        except Exception as e:
            self.get_logger().error(f"MPC step failed: {e}", throttle_duration_sec=2.0)

    def destroy(self):
        self.get_logger().info("Shutting down MPC bridge...")
        self.mpc_controller.destroy()
        super().destroy_node()


# ==========================================================================
# 入口
# ==========================================================================

def main(args=None):
    rclpy.init(args=args)
    bridge = MPCRosBridge(
        robot_file="dual_franka_with_camera.yml",
        optimization_dt=0.025,
        cold_start_iters=100,
        warm_start_iters=200,
    )
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
