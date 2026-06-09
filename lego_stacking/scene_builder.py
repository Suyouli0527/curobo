# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene builder for LEGO stacking: load config, compute brick poses, build Scene."""

from __future__ import annotations

import atexit
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from curobo.scene import Cuboid, Scene
from curobo.types import ContentPath, Pose
from curobo._src.robot.loader.util import load_robot_yaml


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
    baseplate_surface_position: List[float]
    baseplate_grid_size: int
    default_brick_N: int
    default_brick_M: int
    robot_config: str = "dual_franka.yml"
    robot_base_poses: dict = None
    collision_env: Optional[str] = None
    show_robot_spheres: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "SceneConfig":
        with open(path) as f:
            data = json.load(f)

        # Resolve collision_env path relative to the scene_config.json file.
        collision_env = data.get("collision_env", None)
        if collision_env is not None:
            config_dir = Path(path).parent
            env_path = config_dir / collision_env
            if env_path.exists():
                collision_env = str(env_path.resolve())
            else:
                collision_env = str(Path(collision_env).resolve())

        # Support legacy ``baseplate_surface_z`` (scalar) by converting to
        # ``baseplate_surface_position`` (3-D vector).
        if "baseplate_surface_position" in data:
            surface_pos = data["baseplate_surface_position"]
        else:
            z = data.get("baseplate_surface_z", 0.0)
            surface_pos = [0.0, 0.0, z]

        return cls(
            stud_spacing=data["stud_spacing"],
            brick_height=data["brick_height"],
            baseplate_height=data["baseplate_height"],
            baseplate_surface_position=surface_pos,
            baseplate_grid_size=data["baseplate_grid_size"],
            default_brick_N=data["default_brick_N"],
            default_brick_M=data["default_brick_M"],
            robot_config=data.get("robot_config", "dual_franka.yml"),
            robot_base_poses=data.get("robot_base_poses", None),
            collision_env=collision_env,
            show_robot_spheres=data.get("show_robot_spheres", False),
        )


def load_layout(path: str | Path) -> Dict[str, dict]:
    """Load layout.json, return dict of brick_id -> {x, y, z}."""
    with open(path) as f:
        return json.load(f)


def compute_brick_world_pos(
    x: int, y: int, z: int, cfg: SceneConfig
) -> np.ndarray:
    """Convert grid coords (x, y, z) to world position (meters).

    Grid origin is at baseplate top-surface center, offset by
    ``baseplate_surface_position``.
    Brick origin is at its bottom-center.
    """
    pos = cfg.baseplate_surface_position
    bx = pos[0] + (x - (cfg.baseplate_grid_size - 1) / 2.0) * cfg.stud_spacing
    by = pos[1] + (y - (cfg.baseplate_grid_size - 1) / 2.0) * cfg.stud_spacing
    bz = pos[2] + (z - 1) * cfg.brick_height + cfg.brick_height / 2.0
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


def _normalize_collision_dict(scene_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize obstacle dict to match cuRobo's Scene.create expectations.

    Handles legacy ``dims`` on cylinders by converting to ``radius``/``height``.
    """
    if scene_dict is None:
        return {}

    # Convert cylinder dims [radius, height] -> radius/height fields.
    if "cylinder" in scene_dict:
        for name, props in scene_dict["cylinder"].items():
            if "dims" in props and isinstance(props["dims"], list) and len(props["dims"]) == 2:
                props = dict(props)
                props["radius"] = props.pop("dims")[0]
                props["height"] = props.pop("dims")[1]
                scene_dict["cylinder"][name] = props

    # Remove empty obstacle categories so Scene.create doesn't break.
    for key in list(scene_dict.keys()):
        if scene_dict[key] is None or scene_dict[key] == {}:
            del scene_dict[key]

    return scene_dict


def build_scene(scene_cfg: SceneConfig) -> Scene:
    """Build collision Scene with baseplate and optional collision_env."""
    # Start with collision_env if provided.
    scene_dict: Dict[str, Any] = {}
    if scene_cfg.collision_env is not None and os.path.exists(scene_cfg.collision_env):
        with open(scene_cfg.collision_env) as f:
            scene_dict = yaml.safe_load(f) or {}

        # Resolve mesh file paths relative to collision_env.yml directory.
        if "mesh" in scene_dict:
            collision_env_dir = Path(scene_cfg.collision_env).parent
            for name, props in scene_dict["mesh"].items():
                if "file_path" in props:
                    file_path = Path(props["file_path"])
                    if not file_path.is_absolute():
                        props["file_path"] = str((collision_env_dir / file_path).resolve())

    scene_dict = _normalize_collision_dict(scene_dict)

    # Ensure cuboid key exists.
    if "cuboid" not in scene_dict:
        scene_dict["cuboid"] = {}

    # Add baseplate.
    bp_dims = [
        scene_cfg.baseplate_grid_size * scene_cfg.stud_spacing,
        scene_cfg.baseplate_grid_size * scene_cfg.stud_spacing,
        scene_cfg.baseplate_height,
    ]
    # Baseplate origin is at top-surface center, offset by baseplate_surface_position.
    pos = scene_cfg.baseplate_surface_position
    bp_pos = [
        pos[0],
        pos[1],
        pos[2] - scene_cfg.baseplate_height / 2.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    scene_dict["cuboid"]["baseplate"] = {
        "dims": bp_dims,
        "pose": bp_pos,
        "color": [0, 255, 0],
    }

    return Scene.create(scene_dict)


def _quat_to_rpy(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to roll-pitch-yaw (intrinsic XYZ).

    Returns (roll, pitch, yaw) in radians.
    """
    roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return roll, pitch, yaw


#: Map from semantic arm names in scene_config.json to URDF joint names.
DEFAULT_ARM_JOINT_MAP = {
    "right_arm": "world_to_right_arm",
    "left_arm": "world_to_left_arm",
}


def apply_robot_base_poses_to_config(
    robot_config: str,
    robot_base_poses: dict,
    arm_joint_map: Optional[dict] = None,
) -> dict[str, Any]:
    """Load robot config, modify URDF with new base poses, return updated config dict.

    Args:
        robot_config: Path to robot YAML config (e.g., "dual_franka.yml").
        robot_base_poses: Dict mapping arm names to {"position": [x, y, z],
            "quaternion": [w, x, y, z]}.
        arm_joint_map: Optional override for mapping arm names to URDF joint names.

    Returns:
        Modified robot configuration dictionary with updated URDF path.
    """
    if arm_joint_map is None:
        arm_joint_map = DEFAULT_ARM_JOINT_MAP

    content_path = ContentPath(robot_config_file=robot_config)
    robot_data = load_robot_yaml(content_path)

    kin_cfg = robot_data["robot_cfg"]["kinematics"]
    urdf_path = kin_cfg["urdf_path"]

    # Resolve URDF to absolute path.
    # urdf_path in config is relative to asset_root_path (or content root).
    asset_root = content_path.robot_asset_root_path
    if not os.path.isabs(urdf_path):
        # Try asset root first
        abs_urdf = os.path.join(asset_root, urdf_path)
        if not os.path.exists(abs_urdf):
            # Fallback: content root
            from curobo.content import get_assets_path
            abs_urdf = os.path.join(get_assets_path(), urdf_path)
    else:
        abs_urdf = urdf_path

    if not os.path.exists(abs_urdf):
        raise FileNotFoundError(f"URDF not found: {abs_urdf}")

    urdf_dir = os.path.dirname(abs_urdf)

    tree = ET.parse(abs_urdf)
    root = tree.getroot()

    for arm_name, pose in robot_base_poses.items():
        joint_name = arm_joint_map.get(arm_name)
        if joint_name is None:
            continue

        pos = pose["position"]
        quat = pose["quaternion"]
        roll, pitch, yaw = _quat_to_rpy(quat[0], quat[1], quat[2], quat[3])

        # Find and update the joint origin.
        for joint in root.findall("joint"):
            if joint.get("name") == joint_name:
                origin = joint.find("origin")
                if origin is not None:
                    origin.set("xyz", f"{pos[0]} {pos[1]} {pos[2]}")
                    origin.set("rpy", f"{roll} {pitch} {yaw}")
                break

    # Write modified URDF to a temp file in the same directory (so mesh paths resolve).
    urdf_basename = os.path.basename(abs_urdf)
    fd, temp_urdf = tempfile.mkstemp(
        suffix=f"_{urdf_basename}", dir=urdf_dir, prefix="curobo_base_"
    )
    tree.write(temp_urdf, encoding="utf-8", xml_declaration=True)
    os.close(fd)

    # Register cleanup so the temp file is deleted on normal program exit.
    atexit.register(lambda p=temp_urdf: os.path.exists(p) and os.remove(p))

    # Update config to use the temp URDF with absolute path.
    kin_cfg["urdf_path"] = temp_urdf

    return robot_data


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
