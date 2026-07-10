"""抓取 demo — 纯视觉跟随 (无碰撞附着)."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np


class Gripper:
    def __init__(self, obs_info: Dict):
        self._obs = obs_info
        self._name: Optional[str] = None
        self._obs_pos: Optional[np.ndarray] = None
        self._mid_pos: Optional[np.ndarray] = None

    def grasp(self, mid_fk: np.ndarray, target: np.ndarray) -> Optional[str]:
        nearest, best = None, float("inf")
        for n, info in self._obs.items():
            d = float(np.linalg.norm(target - info["pos"]))
            if d < best:
                best, nearest = d, n
        if nearest is None:
            return None
        self._name = nearest
        self._obs_pos = self._obs[nearest]["pos"].copy()
        self._mid_pos = mid_fk.copy()
        self._obs[nearest]["frame"].color = np.array((0, 200, 0), dtype=np.uint8)
        print(f"[grasp] {nearest} (dist={best:.3f}m)")
        return nearest

    def release(self) -> Optional[str]:
        if self._name is None:
            return None
        self._obs[self._name]["frame"].color = np.array(
            self._obs[self._name]["color"], dtype=np.uint8)
        print(f"[release] {self._name}")
        self._name = None
        return None

    def update(self, mid_fk: np.ndarray):
        if self._name is None:
            return
        d = mid_fk - self._mid_pos
        p = self._obs_pos + d
        self._obs[self._name]["frame"].position = p
        self._obs[self._name]["pos"] = p

    @property
    def attached_name(self):
        return self._name
