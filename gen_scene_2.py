#!/usr/bin/env python3
"""生成 myscene_2.yml — 多层货架拣选避障场景 (参数化, 适合强化学习扫参)

场景描述:
  双臂面对一个 1.5m 高 3 层货架, 每层被 2cm 厚钢板分隔成多个 0.35m×0.5m 的方格。
  货架横梁 + 纵向隔板 + 相邻格内干扰物料 构成密集空间格栅障碍。
  机器人需从货架中拣选/放置物料, 规避所有格栅结构。

Usage:
    python gen_scene_2.py -o myscene_2.yml              # 默认参数
    python gen_scene_2.py --num_cells 5 --num_layers 4  # 自定义
"""

import argparse
import yaml
from pathlib import Path

# ── 默认参数 ───────────────────────────────────────────
PARAMS = {
    # ═══ 货架整体 ═══
    "shelf_front_x":  0.50,    # 货架前面板 X 坐标 (m)
    "shelf_depth":    0.50,    # 货架深度 (X方向, m)
    "shelf_height":   1.50,    # 货架总高 (Z方向, m)
    "shelf_y_center": 0.00,    # 货架 Y 中心 (m)

    # ═══ 货架分层 ═══
    "num_layers":     3,       # 层数
    "beam_thickness": 0.02,    # 横梁/隔板厚度 (m)

    # ═══ 货架分格 ═══
    "num_cells":      3,       # 每层格数 (Y方向)
    "cell_width":     0.35,    # 格宽 (Y方向, m)

    # ═══ 干扰物料 ═══
    "distractor_count":      2,    # 干扰物料数量
    "distractor_layer":      0,    # 干扰物料所在层 (0=底层)
    "distractor_cells":      [0, 2],  # 干扰物料所在格索引
    "distractor_fill_ratio": 0.6,     # 物料填充比例
}


def generate(params: dict) -> dict:
    p = PARAMS.copy()
    p.update({k: v for k, v in params.items() if v is not None})

    scene = {"cuboid": {}}

    # ── 计算几何 ──
    num_cells = int(p["num_cells"])
    num_layers = int(p["num_layers"])
    cell_w = p["cell_width"]
    beam_t = p["beam_thickness"]
    shelf_d = p["shelf_depth"]
    shelf_h = p["shelf_height"]
    shelf_fx = p["shelf_front_x"]
    shelf_y0 = p["shelf_y_center"]

    # 货架总宽 (Y): N个格 + (N+1)个隔板
    total_y = num_cells * cell_w + (num_cells + 1) * beam_t
    shelf_y_min = shelf_y0 - total_y / 2

    # X: 货架中心 = front_x + depth/2
    shelf_cx = round(shelf_fx + shelf_d / 2, 10)

    # Z: 每层高度 = (总高 - (层数+1)*横梁厚度) / 层数
    layer_h = (shelf_h - (num_layers + 1) * beam_t) / num_layers

    # ── 1. 水平横梁 (每层上下各一根, 共 N+1 根) ──
    for i in range(num_layers + 1):
        z_bottom = i * (layer_h + beam_t)
        z_center = round(z_bottom + beam_t / 2, 4)
        scene["cuboid"][f"shelf_beam_row{i}"] = {
            "dims": [shelf_d, total_y, beam_t],
            "pose": [shelf_cx, shelf_y0, z_center, 1, 0, 0, 0],
        }

    # ── 2. 纵向隔板 (每格之间, 共 N+1 根) ──
    for i in range(num_cells + 1):
        y_center = round(shelf_y_min + beam_t / 2 + i * (cell_w + beam_t), 10)
        scene["cuboid"][f"shelf_divider_col{i}"] = {
            "dims": [shelf_d, beam_t, shelf_h],
            "pose": [shelf_cx, y_center, shelf_h / 2, 1, 0, 0, 0],
        }

    # ── 3. 干扰物料 (在指定格内放置方块) ──
    d_layer = int(p.get("distractor_layer", 0))
    d_cells = p.get("distractor_cells", [])
    d_fill = p.get("distractor_fill_ratio", 0.6)

    for ci in d_cells:
        if ci >= num_cells:
            continue
        # 格子中心 Y
        cell_cy = round(shelf_y_min + beam_t + ci * (cell_w + beam_t) + cell_w / 2, 10)
        # 格子中心 Z (该层内部)
        cell_z_bottom = d_layer * (layer_h + beam_t) + beam_t
        cell_z_center = round(cell_z_bottom + layer_h / 2, 10)

        # 物料尺寸: 略小于格子
        mat_w = round(cell_w * d_fill - 0.01, 10)
        mat_d = round(shelf_d * d_fill - 0.01, 10)
        mat_h = round(layer_h * d_fill - 0.01, 10)
        mat_cx = shelf_cx  # 居中
        mat_cz = cell_z_center

        scene["cuboid"][f"distractor_layer{d_layer}_cell{ci}"] = {
            "dims": [mat_d, mat_w, mat_h],
            "pose": [mat_cx, cell_cy, mat_cz, 1, 0, 0, 0],
        }

    return scene


def main():
    parser = argparse.ArgumentParser(
        description="多层货架拣选避障场景生成器"
    )
    parser.add_argument("--output", "-o", default=None,
                        help="输出文件路径")
    for key, val in PARAMS.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", type=lambda x: x.lower() != 'false',
                                default=val, help=f"{key} (默认: {val})")
        elif isinstance(val, list):
            parser.add_argument(f"--{key}", type=str, default=str(val),
                                help=f"{key} (默认: {val}), 逗号分隔")
        else:
            parser.add_argument(f"--{key}", type=float, default=val,
                                help=f"{key} (默认: {val})")

    args = parser.parse_args()
    params = {k: getattr(args, k) for k in PARAMS}

    # Parse list args
    for k, v in params.items():
        if isinstance(v, str) and v.startswith('['):
            params[k] = [int(x.strip()) for x in v.strip('[]').split(',')]

    scene = generate(params)

    n_obs = len(scene.get("cuboid", {}))
    header = (
        "# ============================================================\n"
        "# myscene_2.yml — 多层货架拣选避障场景\n"
        "# 双臂在移动底座/直线导轨上, 面对 1.5m 高 3 层货架\n"
        "# 货架横梁 + 纵向隔板 + 格内干扰物料 = 密集空间格栅障碍\n"
        "# 生成: python gen_scene_2.py\n"
        f"# 参数: {params}\n"
        f"# 障碍物数量: {n_obs}\n"
        "# ============================================================\n"
    )
    yaml_str = yaml.dump(scene, default_flow_style=False, sort_keys=False, width=120)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(header + "\n" + yaml_str)
        cu_scene = Path(__file__).parent / "curobo/content/configs/scene" / out_path.name
        cu_scene.parent.mkdir(parents=True, exist_ok=True)
        cu_scene.write_text(header + "\n" + yaml_str)
        print(f"✓ Written to {out_path}")
        print(f"✓ Copied to {cu_scene}")
        print(f"  货架: {params['num_layers']}层 × {params['num_cells']}格, {n_obs} 个障碍物")
        print(f"  层高: {(1.5 - (params['num_layers']+1)*0.02)/params['num_layers']:.3f}m, "
              f"总宽: {params['num_cells']*0.35 + (params['num_cells']+1)*0.02:.2f}m")
    else:
        print(header)
        print(yaml_str)


if __name__ == "__main__":
    main()
