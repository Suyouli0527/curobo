# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cost function for maintaining relative pose between two end-effectors (GPU-accelerated)."""

from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Optional

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.wp_tool_pose import RelativePoseDistance
from curobo._src.types.tool_pose import ToolPose
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg


class RelativePoseCost(BaseCost):
    """Cost that maintains a fixed relative pose between two end-effectors.

    Uses Warp kernels for GPU-accelerated computation with axis-angle rotation error.

    The relative pose is computed as:
        - relative_position = q_primary^* ⊙ (p_secondary - p_primary)
        - relative_quaternion = q_primary^* ⊗ q_secondary

    Position cost: 0.5 * w_p * |rel_pos - target_pos|^2
    Rotation cost: w_r * |axis_angle(rel_quat ⊗ target_quat^{-1})|^2

    Gradients are computed w.r.t. both primary and secondary tool poses.
    """

    def __init__(self, config: RelativePoseCostCfg):
        """Initialize the RelativePoseCost.

        Args:
            config: Configuration for the cost.
        """
        self.config: RelativePoseCostCfg = config
        super().__init__(config)

        self.primary_tool_frame = config.primary_tool_frame
        self.secondary_tool_frame = config.secondary_tool_frame
        self.target_relative_pose = config.target_relative_pose
        self.rotation_method = config.rotation_method

        # weight is [position_weight, rotation_weight] — same convention as ToolPoseCost
        # _weight is inherited from BaseCost

        self._position_axes_weight_t = torch.ones(
            3, dtype=self.device_cfg.dtype, device=self.device_cfg.device
        )
        self._rotation_axes_weight_t = torch.ones(
            3, dtype=self.device_cfg.dtype, device=self.device_cfg.device
        )
        self._convergence_tolerance_pos_t = torch.tensor(
            [config.position_tolerance],
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )
        self._convergence_tolerance_rot_t = torch.tensor(
            [config.orientation_tolerance],
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )

        # Cache for batch tensors
        self._position_cost = None
        self._orientation_cost = None

    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs):
        """Setup batch tensors for position and orientation costs."""
        if batch_size != self._batch_size or horizon != self._horizon:
            self._out_distance = torch.zeros(
                (batch_size, horizon, 2),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_position_distance = torch.zeros(
                (batch_size, horizon, 1),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_rotation_distance = torch.zeros(
                (batch_size, horizon, 1),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_position_gradient_primary = torch.zeros(
                (batch_size, horizon, 3),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_position_gradient_secondary = torch.zeros(
                (batch_size, horizon, 3),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_rotation_gradient_primary = torch.zeros(
                (batch_size, horizon, 4),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_rotation_gradient_secondary = torch.zeros(
                (batch_size, horizon, 4),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            super().setup_batch_tensors(batch_size, horizon)

    def forward(
        self,
        current_tool_poses: ToolPose,
        **kwargs,
    ) -> torch.Tensor:
        """Compute relative pose cost using GPU-accelerated Warp kernel.

        Args:
            current_tool_poses: Current poses of all end-effectors.
            **kwargs: Additional arguments (unused).

        Returns:
            Cost tensor of shape [batch_size, horizon, 1]
        """
        # Validate inputs
        if current_tool_poses is None:
            log_and_raise("current_tool_poses must be provided")

        # Get tool frames
        tool_frames = current_tool_poses.tool_frames
        if self.primary_tool_frame not in tool_frames:
            log_and_raise(
                f"primary_tool_frame '{self.primary_tool_frame}' not found in tool_frames"
            )
        if self.secondary_tool_frame not in tool_frames:
            log_and_raise(
                f"secondary_tool_frame '{self.secondary_tool_frame}' not found in tool_frames"
            )

        # Get indices of the two tools
        idx_primary = tool_frames.index(self.primary_tool_frame)
        idx_secondary = tool_frames.index(self.secondary_tool_frame)

        # Extract poses: shape [batch, horizon, 3/4]
        primary_position = current_tool_poses.position[:, :, idx_primary, :]  # [b, h, 3]
        primary_quat = current_tool_poses.quaternion[:, :, idx_primary, :]  # [b, h, 4]
        secondary_position = current_tool_poses.position[:, :, idx_secondary, :]  # [b, h, 3]
        secondary_quat = current_tool_poses.quaternion[:, :, idx_secondary, :]  # [b, h, 4]

        # Call the GPU-accelerated Warp kernel
        # _weight shape is (2,) => [position_weight, rotation_weight]
        cost_distance, pos_distance, rot_distance = RelativePoseDistance.apply(
            primary_position,
            primary_quat,
            secondary_position,
            secondary_quat,
            self.target_relative_pose.position.squeeze(0),
            self.target_relative_pose.quaternion.squeeze(0),
            self._weight[0:1],
            self._weight[1:2],
            self._position_axes_weight_t,
            self._rotation_axes_weight_t,
            self._convergence_tolerance_pos_t,
            self._convergence_tolerance_rot_t,
            self._out_distance,
            self._out_position_distance,
            self._out_rotation_distance,
            self._out_position_gradient_primary,
            self._out_position_gradient_secondary,
            self._out_rotation_gradient_primary,
            self._out_rotation_gradient_secondary,
        )

        # cost_distance shape: [batch, horizon, 2] -> [pos_cost, rot_cost]
        # Kernel already applied position_weight=self._weight[0] and
        # rotation_weight=self._weight[1]; just sum the two components.
        total_cost = cost_distance[..., 0:1] + cost_distance[..., 1:2]

        # Store raw distances for inspection
        self._position_cost = pos_distance
        self._orientation_cost = rot_distance

        return total_cost

    @property
    def position_error(self) -> torch.Tensor:
        """Get the latest geometric position error (unweighted distance)."""
        return self._position_cost

    @property
    def orientation_error(self) -> torch.Tensor:
        """Get the latest geometric orientation error (unweighted angle in radians)."""
        return self._orientation_cost
