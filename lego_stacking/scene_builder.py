# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene builder for LEGO stacking: load config, compute brick poses, build Scene."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from curobo.scene import Cuboid, Scene
from curobo.types import Pose


SCRIPT_DIR = Path(__file__).parent


@dataclass
class BrickConfig:
    """Parsed brick layout entry."""

    brick_id: str
    x: int
    y: int
    z: int
    world_pos: np.ndarray
    world_pose: Pose
    dims: List[float]


@dataclass
class SceneConfig:
    """Geometry parameters from scene_config.json."""

    stud_spacing: float
    brick_height: float
    baseplate_height: float
    baseplate_surface_z: float
    baseplate_grid_size: int
    default_brick_N: int
    default_brick_M: int
    robot_config: str = "dual_franka.yml"
    robot_base_poses: dict = None

    @classmethod
    def load(cls, path: str | Path) -> "SceneConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(
            stud_spacing=data["stud_spacing"],
            brick_height=data["brick_height"],
            baseplate_height=data["baseplate_height"],
            baseplate_surface_z=data["baseplate_surface_z"],
            baseplate_grid_size=data["baseplate_grid_size"],
            default_brick_N=data["default_brick_N"],
            default_brick_M=data["default_brick_M"],
            robot_config=data.get("robot_config", "dual_franka.yml"),
            robot_base_poses=data.get("robot_base_poses", None),
        )


def load_layout(path: str | Path) -> Dict[str, dict]:
    """Load layout.json, return dict of brick_id -> {x, y, z}."""
    with open(path) as f:
        return json.load(f)


def compute_brick_world_pos(
    x: int, y: int, z: int, cfg: SceneConfig
) -> np.ndarray:
    """Convert grid coords (x, y, z) to world position (meters).

    Grid origin is at baseplate top-surface center.
    Brick origin is at its bottom-center.
    """
    bx = (x - (cfg.baseplate_grid_size - 1) / 2.0) * cfg.stud_spacing
    by = (y - (cfg.baseplate_grid_size - 1) / 2.0) * cfg.stud_spacing
    bz = cfg.baseplate_surface_z + (z - 1) * cfg.brick_height + cfg.brick_height / 2.0
    return np.array([bx, by, bz], dtype=np.float64)


def build_bricks(
    layout: Dict[str, dict], scene_cfg: SceneConfig
) -> Dict[str, BrickConfig]:
    """Build BrickConfig dict from layout."""
    bricks = {}
    for bid, entry in layout.items():
        x, y, z = entry["x"], entry["y"], entry["z"]
        pos = compute_brick_world_pos(x, y, z, scene_cfg)
        dims = [
            scene_cfg.default_brick_N * scene_cfg.stud_spacing,
            scene_cfg.default_brick_M * scene_cfg.stud_spacing,
            scene_cfg.brick_height,
        ]
        pose = Pose(
            position=torch.tensor([[pos.tolist()]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device="cuda", dtype=torch.float32),
        )
        bricks[bid] = BrickConfig(
            brick_id=bid, x=x, y=y, z=z, world_pos=pos, world_pose=pose, dims=dims
        )
    return bricks


def build_scene(scene_cfg: SceneConfig) -> Scene:
    """Build collision Scene with baseplate only (bricks are collision-free)."""
    bp_dims = [
        scene_cfg.baseplate_grid_size * scene_cfg.stud_spacing,
        scene_cfg.baseplate_grid_size * scene_cfg.stud_spacing,
        scene_cfg.baseplate_height,
    ]
    # Baseplate origin is at top-surface center
    bp_pos = [
        0.0,
        0.0,
        scene_cfg.baseplate_surface_z - scene_cfg.baseplate_height / 2.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    return Scene(cuboid=[Cuboid(name="baseplate", dims=bp_dims, pose=bp_pos)])


def create_scene_and_bricks(
    scene_config_path: str | Path = None,
    layout_path: str | Path = None,
) -> tuple[Scene, SceneConfig, Dict[str, BrickConfig]]:
    """Convenience: load configs, build Scene and bricks dict."""
    if scene_config_path is None:
        scene_config_path = SCRIPT_DIR / "config" / "scene_config.json"
    if layout_path is None:
        layout_path = SCRIPT_DIR / "config" / "layout.json"

    scene_cfg = SceneConfig.load(scene_config_path)
    layout = load_layout(layout_path)
    bricks = build_bricks(layout, scene_cfg)
    scene = build_scene(scene_cfg)
    return scene, scene_cfg, bricks
