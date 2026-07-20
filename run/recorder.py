"""MPC 数据记录与画图."""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class MPCRecorder:
    """收集每步 solve_time / scene_dist / self_dist / rel_pose_err，退出后画曲线保存."""

    def __init__(self, out_dir: Optional[str] = None):
        self.solve_times: List[float] = []
        self.scene_dists: List[float] = []
        self.self_dists: List[float] = []
        self.rel_pos_errs: List[float] = []
        self.rel_rot_errs: List[float] = []
        self.obj_vals: List[float] = []
        self.grasp_steps: List[int] = []

        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(__file__), "plots")
        self._out_dir = out_dir

    def record(self, solve_time: float, scene_dist: float, self_dist: float,
               rel_pos_err: float = 0.0, rel_rot_err: float = 0.0,
               obj_val: float = 0.0) -> None:
        """记录一步的数据."""
        self.solve_times.append(solve_time)
        self.scene_dists.append(scene_dist)
        self.self_dists.append(self_dist)
        self.rel_pos_errs.append(rel_pos_err)
        self.rel_rot_errs.append(rel_rot_err)
        self.obj_vals.append(obj_val)

    def mark_grasp(self) -> None:
        """标记当前 step 处发生抓取."""
        self.grasp_steps.append(len(self.solve_times))

    @property
    def steps(self) -> int:
        return len(self.solve_times)

    def _plot(self, start: int, end: int, out_path: str, title_suffix: str = "") -> Optional[str]:
        """画曲线 [start, end) 范围并保存."""
        sc = np.array(self.scene_dists[start:end])
        st = np.array(self.solve_times[start:end])
        rp = np.array(self.rel_pos_errs[start:end])
        rr = np.array(self.rel_rot_errs[start:end])
        ov = np.array(self.obj_vals[start:end])

        if len(st) == 0:
            return None

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        LEFT_COLOR = "#d62728"
        RIGHT_COLOR = "#1f77b4"

        axes[0].plot(st, color=LEFT_COLOR)
        ax0r = axes[0].twinx()
        ax0r.plot(ov, color=RIGHT_COLOR, linewidth=0.3, alpha=0.7)
        ax0r.set_ylabel("Obj Value", color=RIGHT_COLOR)
        ax0r.tick_params(axis="y", labelcolor=RIGHT_COLOR)
        axes[0].set_ylabel("Solve Time (s)", color=LEFT_COLOR)
        axes[0].tick_params(axis="y", labelcolor=LEFT_COLOR)
        axes[0].set_title(f"MPC Performance{title_suffix}  |  steps={len(st)}  |  "
                          f"solve avg={st.mean():.4f}s  max={st.max():.4f}s  |  "
                          f"obj min={ov.min():.1f}  max={ov.max():.1f}")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(sc, color=LEFT_COLOR, linewidth=0.5)
        axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("Min Scene Collision Distance (m)", color=LEFT_COLOR)
        axes[1].tick_params(axis="y", labelcolor=LEFT_COLOR)
        axes[1].set_title(f"min={sc.min()*1000:.1f}mm  "
                          f"avg={sc.mean()*1000:.1f}mm  "
                          f"max={sc.max()*1000:.1f}mm")
        axes[1].grid(True, alpha=0.3)

        ax3 = axes[2]
        ax3.plot(rp, color=LEFT_COLOR, linewidth=0.5, label="pos (mm)")
        ax3.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax3.set_ylabel("Rel Pos Err (mm)", color=LEFT_COLOR)
        ax3.tick_params(axis="y", labelcolor=LEFT_COLOR)
        ax3.set_xlabel("MPC Step")
        ax3.set_title(f"rel pos_err  min={rp.min():.1f}mm  avg={rp.mean():.1f}mm  max={rp.max():.1f}mm  ||  "
                      f"rel rot_err  min={rr.min():.2f}deg  avg={rr.mean():.2f}deg  max={rr.max():.2f}deg")
        ax3.grid(True, alpha=0.3)

        ax3r = ax3.twinx()
        ax3r.plot(rr, color=RIGHT_COLOR, linewidth=0.5, label="rot (deg)")
        ax3r.set_ylabel("Rel Rot Err (deg)", color=RIGHT_COLOR)
        ax3r.tick_params(axis="y", labelcolor=RIGHT_COLOR)

        lines = [ax3.get_lines()[0], ax3r.get_lines()[0]]
        ax3.legend(lines, [l.get_label() for l in lines], loc="upper right")

        # 抓取线: 用绝对 step 号减 start 得到相对位置
        for gs in self.grasp_steps:
            rel_gs = gs - start
            if 0 <= rel_gs < end - start:
                for ax in axes:
                    ax.axvline(x=rel_gs, color="green", linestyle="--",
                               linewidth=1.0, alpha=0.7)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path

    def save(self) -> Optional[str]:
        """画全部曲线保存到 plots/."""
        if not self.solve_times:
            return None
        os.makedirs(self._out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(self._out_dir, f"mpc_curves_{ts}.png")
        result = self._plot(0, len(self.solve_times), out)
        if result:
            sc = np.array(self.scene_dists)
            st = np.array(self.solve_times)
            print(f"\nCurves saved to {result}")
            print(f"  Steps: {len(st)}  |  "
                  f"Solve time: avg={st.mean():.4f}s  max={st.max():.4f}s  |  "
                  f"Min scene dist: {sc.min()*1000:.1f}mm")
        return result

    def save_attach(self) -> Optional[str]:
        """画抓取后曲线保存到 plots/attach/."""
        if not self.solve_times or not self.grasp_steps:
            return None
        start = self.grasp_steps[0]
        end = len(self.solve_times)
        out_dir = os.path.join(self._out_dir, "attach")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(out_dir, f"mpc_curves_attach_{ts}.png")
        result = self._plot(start, end, out, title_suffix=" (post-attach)")
        if result:
            print(f"Attach curves saved to {result}")
        return result
