"""抓取模块 — 视觉跟随 + 碰撞附着."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

class Gripper:
    def __init__(self, obs_info: Dict, controller=None):
        self._obs, self._controller = obs_info, controller
        self._name: Optional[str] = None
        self._orig_pos: Optional[np.ndarray] = None

    def grasp(self, target: np.ndarray, left_frame=None, right_frame=None) -> Optional[str]:
        nearest, best = None, float("inf")
        for n, info in self._obs.items():
            d = float(np.linalg.norm(target - info["pos"]))
            if d < best: best, nearest = d, n
        if nearest is None: return None
        if self._controller is not None:
            if not self._controller.grasp_obstacle(nearest,
                                                    left_frame=left_frame,
                                                    right_frame=right_frame): return None
        self._name = nearest
        self._orig_pos = self._obs[nearest]["pos"].copy()
        self._obs[nearest]["frame"].visible = False
        return nearest

    def release(self) -> Optional[str]:
        if self._name is None: return None
        if self._controller is not None: self._controller.release_obstacle()
        self._obs[self._name]["frame"].visible = True
        self._obs[self._name]["frame"].color = np.array(self._obs[self._name]["color"], dtype=np.uint8)
        self._obs[self._name]["frame"].position = self._orig_pos
        n, self._name = self._name, None
        self._orig_pos = None
        return n

    def update(self, pos: np.ndarray):
        if self._name is None: return
        self._obs[self._name]["frame"].position = pos

    @property
    def attached_name(self) -> Optional[str]: return self._name
