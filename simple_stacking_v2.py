# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cube stacking demo using cuRobo v2 with Viser visualization.

A Franka robot picks up cubes from a 3x3 grid and stacks them into a tower.
Each pick and each place is planned and executed separately, advanced
manually by the user via the "Next Step" button.

Run:
    python simple_stacking_v2.py

Then open http://localhost:8080 in your browser. Click "Next Step" to
plan and execute the next pick or place action.
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
from curobo._src.types.content_path import ContentPath
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer

# Cube configuration
CUBE_SIZE = np.array([0.045, 0.045, 0.07])
CUBE_Z_CENTER = 0.1

CUBE_POSITIONS = [
    [0.50, 0.0, CUBE_Z_CENTER],
    [0.50, -0.20, CUBE_Z_CENTER],
    [0.50, 0.20, CUBE_Z_CENTER],
    [0.30, -0.20, CUBE_Z_CENTER],
    [0.30, 0.0, CUBE_Z_CENTER],
    [0.30, 0.20, CUBE_Z_CENTER],
    [0.70, -0.20, CUBE_Z_CENTER],
    [0.70, 0.0, CUBE_Z_CENTER],
    [0.70, 0.20, CUBE_Z_CENTER],
]

GRASP_OFFSET = CUBE_SIZE[2] / 2.0 + 0.092
TABLE_DIMS = [1.4, 1.4, 0.05]
TABLE_POSE = [0.0, 0.0, -0.05, 1, 0, 0, 0.0]
TOWER_POSITION = np.array([0.50, 0.0, CUBE_Z_CENTER])


def create_scene_with_cubes(
    cube_positions: List[List[float]], cube_names: List[str]
) -> Scene:
    """Create a Scene with table and cubes for collision checking."""
    cuboids = [Cuboid(name="table", dims=TABLE_DIMS, pose=TABLE_POSE)]
    for name, pos in zip(cube_names, cube_positions):
        cuboids.append(
            Cuboid(name=name, dims=CUBE_SIZE.tolist(), pose=pos + [1, 0, 0, 0])
        )
    return Scene(cuboid=cuboids)


def create_grasp_pose(cube_position: np.ndarray) -> Pose:
    """Create a grasp pose above a cube, with gripper pointing downward."""
    grasp_pos = cube_position.copy()
    grasp_pos[2] += GRASP_OFFSET
    return Pose(
        position=torch.tensor([grasp_pos.tolist()], device="cuda", dtype=torch.float32),
        # rot_y_180 quaternion so the Franka gripper z-axis points downward
        quaternion=torch.tensor([[0.0, 0.0, -1.0, 0.0]], device="cuda", dtype=torch.float32),
    )


class StackingController:
    """Step-by-step stacking controller with user confirmation per step."""

    def __init__(self, planner: MotionPlanner, viser_viz: ViserVisualizer):
        self.planner = planner
        self.viser_viz = viser_viz
        self.server = viser_viz._server

        self.cube_names = [f"cube_{i}" for i in range(len(CUBE_POSITIONS))]
        self.cube_positions = [np.array(p) for p in CUBE_POSITIONS]
        self.stacked_cubes: List[int] = []
        self.remaining_cubes: List[int] = list(range(len(CUBE_POSITIONS)))
        self.held_cube_idx: Optional[int] = None
        self.tower_height = 0.0
        self.cube_handles: Dict[str, any] = {}

        # State machine:
        #   "READY_TO_PICK" - next action will pick the next cube
        #   "READY_TO_PLACE" - holding a cube, next action will place it
        #   "DONE" - all cubes stacked
        self.phase = "READY_TO_PICK"

        # Joint state after the last executed action (used as start for next plan)
        self.last_joint_state: Optional[JointState] = None

        # Prevent re-entering during execution
        self.is_busy = False

        self._init_viser_cubes()
        self._init_gui()

    # ------------------------------------------------------------------ #
    #  Initialization                                                    #
    # ------------------------------------------------------------------ #

    def _init_viser_cubes(self):
        for i, pos in enumerate(self.cube_positions):
            name = self.cube_names[i]
            self.cube_handles[name] = self.server.scene.add_box(
                name=f"/cubes/{name}",
                dimensions=tuple(CUBE_SIZE),
                position=tuple(pos),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                color=(100, 150, 200),
            )

    def _init_gui(self):
        with self.server.gui.add_folder("Stacking Control"):
            self.next_btn = self.server.gui.add_button("Next Step", color="green")
            self.reset_btn = self.server.gui.add_button("Reset", color="red")
            self.status_text = self.server.gui.add_text(
                "Status", initial_value="Ready - click 'Next Step' to pick cube 0"
            )
        self.next_btn.on_click(lambda _: self._on_next_step())
        self.reset_btn.on_click(lambda _: self._on_reset())

    # ------------------------------------------------------------------ #
    #  GUI callbacks                                                     #
    # ------------------------------------------------------------------ #

    def _on_next_step(self):
        if self.is_busy or self.phase == "DONE":
            return
        threading.Thread(target=self._execute_next_step, daemon=True).start()

    def _on_reset(self):
        # Reset state (mark busy briefly to block Next during reset)
        self.is_busy = True
        try:
            self.stacked_cubes = []
            self.remaining_cubes = list(range(len(CUBE_POSITIONS)))
            self.held_cube_idx = None
            self.tower_height = 0.0
            self.cube_positions = [np.array(p) for p in CUBE_POSITIONS]
            self.last_joint_state = None
            self.phase = "READY_TO_PICK"

            for i, pos in enumerate(self.cube_positions):
                name = self.cube_names[i]
                self.cube_handles[name].position = tuple(pos)
                self.cube_handles[name].visible = True

            self.viser_viz.reset_robot()
            scene = create_scene_with_cubes(self.cube_positions, self.cube_names)
            self.planner.update_world(scene)
            self.status_text.value = "Reset - click 'Next Step' to pick cube 0"
        finally:
            self.is_busy = False

    # ------------------------------------------------------------------ #
    #  Trajectory animation                                              #
    # ------------------------------------------------------------------ #

    def _animate_trajectory(
        self,
        trajectory: JointState,
        last_tstep: Optional[torch.Tensor] = None,
        dt: float = 0.02,
    ) -> Optional[JointState]:
        if trajectory is None or trajectory.position is None:
            return None

        if last_tstep is not None:
            trajectory = trim_joint_state_trajectory(
                trajectory, 0, int(last_tstep.item())
            )

        n_waypoints = trajectory.position.shape[-2]
        last_js = None
        for i in range(n_waypoints):
            # interpolated_trajectory is 4D (batch, seed, time, dof)
            if trajectory.position.ndim == 4:
                pos = trajectory.position[0, 0, i, :].unsqueeze(0)
            else:
                pos = trajectory.position[0, i, :].unsqueeze(0)
            js = JointState.from_position(pos, joint_names=trajectory.joint_names)
            self.viser_viz.set_joint_state(js)
            last_js = js
            if self.held_cube_idx is not None:
                self._update_held_cube(js)
            time.sleep(dt)
        return last_js

    def _update_held_cube(self, joint_state: JointState):
        # Reindex joint state to match the kinematics' expected joint order
        # (trajectory joint_names may differ in ordering from kinematics).
        js = joint_state.clone()
        js.reindex(self.planner.kinematics.config.kinematics_config.joint_names)
        kin_state = self.planner.compute_kinematics(js)
        ee_pose = kin_state.tool_poses[self.planner.tool_frames[0]]
        ee_pos = ee_pose.position.cpu().squeeze().numpy()
        cube_pos = ee_pos.copy()
        cube_pos[2] -= GRASP_OFFSET - CUBE_SIZE[2] / 2.0
        self.cube_handles[self.cube_names[self.held_cube_idx]].position = tuple(cube_pos)

    # ------------------------------------------------------------------ #
    #  Collision scene helpers                                           #
    # ------------------------------------------------------------------ #

    def _disable_cube(self, cube_idx: int):
        self.planner.scene_collision_checker.enable_obstacle(
            self.cube_names[cube_idx], enable=False
        )

    def _enable_cube(self, cube_idx: int, position: np.ndarray):
        name = self.cube_names[cube_idx]
        pose = Pose(
            position=torch.tensor([position.tolist()], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        )
        self.planner.scene_collision_checker.update_obstacle_pose(name, pose)
        self.planner.scene_collision_checker.enable_obstacle(name, enable=True)

    # ------------------------------------------------------------------ #
    #  Planning and execution                                            #
    # ------------------------------------------------------------------ #

    def _get_current_active_js(self) -> JointState:
        """Return current joint state as active-only (arm joints, no gripper).

        plan_grasp expects active joint state (7 dof for Franka), not the
        full joint state (9 dof including finger joints).
        """
        if self.last_joint_state is None:
            js = self.planner.default_joint_state.clone()
        else:
            js = self.last_joint_state.clone()
        if js.position.ndim == 1:
            js = js.unsqueeze(0)
        # Extract active joints (excludes locked/finger joints)
        return self.planner.kinematics.get_active_js(js)

    def _plan_and_execute_grasp(
        self, target_position: np.ndarray, desc: str
    ) -> Optional[JointState]:
        """Plan a grasp motion (approach -> grasp -> lift) and animate it."""
        grasp_pose = create_grasp_pose(target_position)
        goal = GoalToolPose.from_poses(
            {self.planner.tool_frames[0]: grasp_pose},
            num_goalset=1,
        )
        current_js = self._get_current_active_js()

        print(f"  [PLAN]  {desc}  target={target_position.tolist()}")
        self.status_text.value = f"Planning {desc}..."
        t0 = time.time()
        result = self.planner.plan_grasp(
            grasp_poses=goal,
            current_state=current_js,
            grasp_approach_offset=-0.12,
            grasp_lift_offset=-0.10,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=True,
            grasp_lift_in_tool_frame=True,
        )
        plan_time = time.time() - t0

        if not result.success.any():
            print(f"  [FAIL]  {desc} planning failed ({plan_time:.2f}s): {result.status}")
            self.status_text.value = f"{desc} planning failed: {result.status}"
            return None

        print(f"  [OK]    {desc} planned in {plan_time:.2f}s, executing trajectory...")
        self.status_text.value = f"Executing {desc}..."
        t0 = time.time()
        final_js = None
        for traj, last_t in [
            (result.approach_interpolated_trajectory, result.approach_interpolated_last_tstep),
            (result.grasp_interpolated_trajectory, result.grasp_interpolated_last_tstep),
            (result.lift_interpolated_trajectory, result.lift_interpolated_last_tstep),
        ]:
            if traj is not None:
                final_js = self._animate_trajectory(traj, last_t)
        exec_time = time.time() - t0
        print(f"  [DONE]  {desc} executed in {exec_time:.2f}s")
        return final_js

    def _execute_next_step(self):
        """Execute one step (pick or place) of the stacking sequence."""
        self.is_busy = True
        try:
            if self.phase == "READY_TO_PICK":
                self._do_pick()
            elif self.phase == "READY_TO_PLACE":
                self._do_place()
        finally:
            self.is_busy = False

    def _do_pick(self):
        """Pick up the next remaining cube."""
        if not self.remaining_cubes:
            self.phase = "DONE"
            self.status_text.value = "Stacking complete!"
            return

        pick_idx = self.remaining_cubes[0]
        pick_pos = self.cube_positions[pick_idx]
        print(f"\n=== PICK cube_{pick_idx} from {pick_pos.tolist()} ===")
        self.status_text.value = f"Picking cube {pick_idx}..."

        # Disable target cube to avoid wrist-link collision
        self._disable_cube(pick_idx)

        pick_final_js = self._plan_and_execute_grasp(pick_pos, desc=f"pick cube_{pick_idx}")
        if pick_final_js is None:
            print(f"=== PICK cube_{pick_idx} FAILED ===\n")
            return

        # "Pick up": hide cube visual, follow gripper
        self.held_cube_idx = pick_idx
        self.cube_handles[self.cube_names[pick_idx]].visible = False
        self.remaining_cubes.remove(pick_idx)
        self.last_joint_state = pick_final_js
        self.phase = "READY_TO_PLACE"
        print(f"=== PICK cube_{pick_idx} COMPLETE ({len(self.remaining_cubes)} remaining) ===\n")
        self.status_text.value = (
            f"Picked cube {pick_idx}. Click 'Next Step' to place it on the tower."
        )

    def _do_place(self):
        """Place the held cube onto the tower."""
        if self.held_cube_idx is None:
            self.status_text.value = "No cube held - this shouldn't happen"
            return

        pick_idx = self.held_cube_idx
        place_pos = TOWER_POSITION.copy()
        place_pos[2] += self.tower_height
        print(f"\n=== PLACE cube_{pick_idx} at {place_pos.tolist()} (tower h={self.tower_height:.3f}m) ===")
        self.status_text.value = f"Placing cube {pick_idx} on tower..."

        # Disable stacked cubes at tower to avoid collision during place
        for stacked_idx in self.stacked_cubes:
            self._disable_cube(stacked_idx)

        place_final_js = self._plan_and_execute_grasp(
            place_pos, desc=f"place cube_{pick_idx}"
        )

        # Re-enable stacked cubes regardless of success
        for stacked_idx in self.stacked_cubes:
            self._enable_cube(stacked_idx, self.cube_positions[stacked_idx])

        if place_final_js is None:
            print(f"=== PLACE cube_{pick_idx} FAILED ===\n")
            return

        # "Place": move cube to final position, show in scene
        final_place_pos = place_pos.copy()
        final_place_pos[2] -= GRASP_OFFSET - CUBE_SIZE[2] / 2.0
        self.held_cube_idx = None
        self.cube_positions[pick_idx] = final_place_pos
        self._enable_cube(pick_idx, final_place_pos)
        self.cube_handles[self.cube_names[pick_idx]].position = tuple(final_place_pos)
        self.cube_handles[self.cube_names[pick_idx]].visible = True

        self.stacked_cubes.append(pick_idx)
        self.tower_height += CUBE_SIZE[2]
        self.last_joint_state = place_final_js
        print(f"=== PLACE cube_{pick_idx} COMPLETE (stacked {len(self.stacked_cubes)}/{len(CUBE_POSITIONS)}) ===\n")

        if self.remaining_cubes:
            self.phase = "READY_TO_PICK"
            next_idx = self.remaining_cubes[0]
            self.status_text.value = (
                f"Placed cube {pick_idx} ({len(self.stacked_cubes)}/{len(CUBE_POSITIONS)}). "
                f"Click 'Next Step' to pick cube {next_idx}."
            )
        else:
            self.phase = "DONE"
            print("=== ALL CUBES STACKED ===\n")
            self.status_text.value = "Stacking complete!"


def main():
    parser = argparse.ArgumentParser(description="Cube Stacking with cuRobo v2")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--robot", type=str, default="franka.yml", help="Robot config file")
    args = parser.parse_args()

    print("Initializing MotionPlanner...")
    cube_names = [f"cube_{i}" for i in range(len(CUBE_POSITIONS))]
    scene = create_scene_with_cubes(CUBE_POSITIONS, cube_names)

    config = MotionPlannerCfg.create(
        robot=args.robot,
        scene_model="collision_table.yml",
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
        content_path=ContentPath(robot_config_file=args.robot),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_robot_to_scene=True,
        add_control_frames=False,
        visualize_robot_spheres=False,
    )

    controller = StackingController(planner, viser_viz)

    print(f"\nViser running at http://localhost:{args.port}")
    print("Click 'Next Step' in the Viser GUI to plan and execute each pick/place.")
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
