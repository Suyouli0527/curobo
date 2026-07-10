"""可视化层 — Viser 初始化, 场景障碍物渲染, 控制框管理."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import yaml
from pathlib import Path
import sys

from curobo.types import ContentPath
from curobo.viewer import ViserVisualizer

from run.utils import numpy_to_pose, pose_to_numpy

# 颜色渲染
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

CUROBO_WS = Path(__file__).parent.parent
MY_SCENE_DIR = Path(__file__).parent / "myScene"
SCENE_CONFIG_DIR = CUROBO_WS / "curobo" / "content" / "configs" / "scene"


def load_scene_yaml(scene_name: str) -> dict:
    """加载场景 YAML, 优先级: myScene/ > curobo内置 > 根目录."""
    for base_dir in (MY_SCENE_DIR, SCENE_CONFIG_DIR, CUROBO_WS):
        path = base_dir / scene_name
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
    print(f"✗ Scene file not found: {scene_name}")
    sys.exit(1)


def _get_color(name: str) -> Tuple[int, int, int]:
    for pfx, c in _COLOR_MAP.items():
        if name.startswith(pfx):
            return c
    return (180, 180, 50)


def setup_viser(robot_file: str, port: int = 8080):
    """初始化 Viser, 返回 (viser_viz, server)."""
    print("Initializing Viser...")
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_robot_to_scene=True,
        add_control_frames=False,
        visualize_robot_spheres=False,
    )
    return viser_viz, viser_viz._server  


def render_scene_obstacles(server, scene_dict: dict) -> dict:
    """将 YAML 场景中的障碍物渲染到 Viser, 返回 {name: {pos, color, frame}}."""
    from curobo.scene import Scene
    scene = Scene.create(scene_dict)
    info = {}
    if hasattr(scene, 'cuboid') and scene.cuboid:
        for obs in scene.cuboid:
            c = _get_color(obs.name)
            f = server.scene.add_box(
                f"/obstacles/{obs.name}",
                dimensions=np.array(obs.dims),
                position=np.array(obs.pose[:3]),
                wxyz=np.array(obs.pose[3:]),
                color=np.array(c, dtype=np.uint8),
            )
            info[obs.name] = {"pos": np.array(obs.pose[:3]), "color": c, "frame": f}
    if hasattr(scene, 'sphere') and scene.sphere:
        for obs in scene.sphere:
            c = _get_color(obs.name)
            f = server.scene.add_icosphere(
                f"/obstacles/{obs.name}",
                radius=obs.radius,
                position=np.array(obs.pose[:3]),
                color=np.array(c, dtype=np.uint8),
            )
            info[obs.name] = {"pos": np.array(obs.pose[:3]), "color": c, "frame": f}
    return info


def create_control_frames(server, left_pose, right_pose):
    """创建两个末端控制框, 返回 (left_frame, right_frame)."""
    left_pos, left_wxyz = left_pose
    right_pos, right_wxyz = right_pose
    left_frame = server.scene.add_transform_controls(
        name="/target/left", scale=0.15,
        position=left_pos, wxyz=left_wxyz,
    )
    right_frame = server.scene.add_transform_controls(
        name="/target/right", scale=0.15,
        position=right_pos, wxyz=right_wxyz,
    )
    return left_frame, right_frame
