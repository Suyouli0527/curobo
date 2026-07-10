"""数学工具 — 四元数运算 + 位姿转换 (torch / numpy)."""
#在添加约束时用到的数学工具
from __future__ import annotations

import numpy as np
import torch

from curobo.types import Pose


# ═══════════════════════════════════════════════════════
# 四元数 (torch)
# ═══════════════════════════════════════════════════════


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate (inverse for unit quaternions). q = [w, x, y, z]."""
    c = q.clone()
    c[..., 1:] *= -1
    return c


def quat_rotate_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by unit quaternion q: v_rot = q * v * q^{-1}."""
    q_w, q_x, q_y, q_z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    v_x, v_y, v_z = v[..., 0], v[..., 1], v[..., 2]
    t_x = 2.0 * (q_y * v_z - q_z * v_y)
    t_y = 2.0 * (q_z * v_x - q_x * v_z)
    t_z = 2.0 * (q_x * v_y - q_y * v_x)
    return torch.stack([
        v_x + q_w * t_x + (q_y * t_z - q_z * t_y),
        v_y + q_w * t_y + (q_z * t_x - q_x * t_z),
        v_z + q_w * t_z + (q_x * t_y - q_y * t_x),
    ], dim=-1)


# ═══════════════════════════════════════════════════════
# 四元数 (numpy)
# ═══════════════════════════════════════════════════════


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product in numpy."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + w2 * x1 + y1 * z2 - y2 * z1,
        w1 * y2 + w2 * y1 + z1 * x2 - z2 * x1,
        w1 * z2 + w2 * z1 + x1 * y2 - x2 * y1,
    ], dtype=np.float64)


def quat_rotate_vector_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    v_x, v_y, v_z = v[0], v[1], v[2]
    t_x = 2.0 * (q_y * v_z - q_z * v_y)
    t_y = 2.0 * (q_z * v_x - q_x * v_z)
    t_z = 2.0 * (q_x * v_y - q_y * v_x)
    return np.array([
        v_x + q_w * t_x + (q_y * t_z - q_z * t_y),
        v_y + q_w * t_y + (q_z * t_x - q_x * t_z),
        v_z + q_w * t_z + (q_x * t_y - q_y * t_x),
    ], dtype=np.float64)


# ═══════════════════════════════════════════════════════
# 位姿转换
# ═══════════════════════════════════════════════════════


def pose_to_numpy(position: torch.Tensor, quaternion: torch.Tensor):
    return (
        position.detach().cpu().squeeze().numpy().astype(np.float64),
        quaternion.detach().cpu().squeeze().numpy().astype(np.float64),
    )


def numpy_to_pose(pos: np.ndarray, wxyz: np.ndarray, device: torch.device) -> Pose:
    return Pose(
        position=torch.tensor(pos, dtype=torch.float32, device=device).view(1, 1, 1, 3),
        quaternion=torch.tensor(wxyz, dtype=torch.float32, device=device).view(1, 1, 1, 4),
    )
