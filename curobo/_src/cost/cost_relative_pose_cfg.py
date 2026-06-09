# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for relative pose cost between two tool frames."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional, Type

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_relative_pose import RelativePoseCost
from curobo._src.types.pose import Pose


@dataclass
class RelativePoseCostCfg(BaseCostCfg):
    """Configuration for relative pose cost between two end-effectors.

    This cost maintains a fixed relative pose (position and orientation) between
    two tool frames, useful for dual-arm manipulation tasks like grasping an object.

    The ``weight`` follows the same convention as ``ToolPoseCostCfg``:
    ``[position_weight, rotation_weight]``.
    """

    #: Class type of the cost
    class_type: Type[RelativePoseCost] = RelativePoseCost

    #: Name of the primary end-effector (e.g., 'tool0')
    primary_tool_frame: Optional[str] = None

    #: Name of the secondary end-effector (e.g., 'tool1')
    secondary_tool_frame: Optional[str] = None

    #: Target relative pose of secondary w.r.t. primary frame.
    #: position = [x, y, z], quaternion = [w, x, y, z]
    target_relative_pose: Optional[Pose] = None

    #: Rotation method: "quat" or "6d" for computing rotation distance
    rotation_method: str = "quat"

    #: Position convergence tolerance (m). When the position error is below
    #: this threshold, the kernel sets the position cost to 0.
    position_tolerance: float = 0.0

    #: Orientation convergence tolerance (rad). When the rotation error is below
    #: this threshold, the kernel sets the rotation cost to 0.
    orientation_tolerance: float = 0.0

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.primary_tool_frame is None or self.secondary_tool_frame is None:
            raise ValueError(
                "Both primary_tool_frame and secondary_tool_frame must be specified"
            )

        if self.primary_tool_frame == self.secondary_tool_frame:
            raise ValueError(
                "primary_tool_frame and secondary_tool_frame must be different"
            )

        if self.target_relative_pose is None:
            self.target_relative_pose = Pose(
                position=torch.zeros(3, dtype=self.device_cfg.dtype, device=self.device_cfg.device),
                quaternion=torch.tensor(
                    [1.0, 0.0, 0.0, 0.0],
                    dtype=self.device_cfg.dtype,
                    device=self.device_cfg.device,
                ),
            )

        super().__post_init__()

    def clone(self):
        """Create a deep copy of this configuration."""
        return RelativePoseCostCfg(
            weight=self.weight.clone(),
            device_cfg=self.device_cfg,
            convert_to_binary=self.convert_to_binary,
            primary_tool_frame=self.primary_tool_frame,
            secondary_tool_frame=self.secondary_tool_frame,
            target_relative_pose=self.target_relative_pose.clone(),
            rotation_method=self.rotation_method,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
        )
