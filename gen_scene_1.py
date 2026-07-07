#!/usr/bin/env python3
"""生成 myscene_1.yml — 长条物件搬运避障场景 (参数化, 适合强化学习扫参)

场景描述:
  类似汽车电池模组长梁 / 轨道装备零件 / 铝型材搬运。
  长条形零件(200cm×7cm×7cm)贴地放置, 两排高柱形成狭窄通道,
  双机械臂需协同抬起零件穿越通道。

Usage:
    python gen_scene_1.py                               # 默认参数
    python gen_scene_1.py --long_length 2.5             # 覆盖单参数
    python gen_scene_1.py --row2_enabled false          # 禁用第2排
    python gen_scene_1.py -o custom.yml                 # 指定输出文件
"""

import argparse
import yaml
from pathlib import Path

# ── 默认参数 ───────────────────────────────────────────
PARAMS = {
    # ═══ 长条形零件 ═══
    "long_length":    2.00,    # X 方向长度 (m)
    "long_width":     0.07,    # Y 方向宽度 (m)
    "long_height":    0.07,    # Z 方向高度 (m)
    "long_x_near":    0.10,    # 零件近边距机器人原点 (m)

    # ═══ 第1排高柱 (近排) ═══
    "row1_enabled":   True,    # 是否启用第1排
    "row1_x":         0.50,    # 柱排 X 坐标 (m)
    "row1_pillar_x":  0.10,    # 单柱 X 尺寸 (m)
    "row1_pillar_y":  0.10,    # 单柱 Y 尺寸 (m)
    "row1_height":    0.50,    # 柱高度 (m)
    "row1_spacing":   0.30,    # 柱 Y 间距 (m)
    "row1_start_y":  -0.60,    # 首柱 Y 坐标 (m)
    "row1_count":     5,       # 柱数量

    # ═══ 第2排高柱 (远排, 形成通道) ═══
    "row2_enabled":   True,    # 是否启用第2排
    "row2_x":         0.80,    # 柱排 X 坐标 (m)
    "row2_pillar_x":  0.10,    # 单柱 X 尺寸 (m)
    "row2_pillar_y":  0.10,    # 单柱 Y 尺寸 (m)
    "row2_height":    0.50,    # 柱高度 (m)
    "row2_spacing":   0.30,    # 柱 Y 间距 (m)
    "row2_start_y":  -0.45,    # 首柱 Y 坐标 (m, 与第1排错开)
    "row2_count":     5,       # 柱数量
}


def _add_row(scene: dict, row_prefix: str, p: dict, row_key: str):
    """在场景中添加一排高柱。"""
    if not p.get(f"{row_key}_enabled", True):
        return

    px = p[f"{row_key}_pillar_x"]
    py = p[f"{row_key}_pillar_y"]
    ph = p[f"{row_key}_height"]
    row_x = p[f"{row_key}_x"]
    spacing = p[f"{row_key}_spacing"]
    start_y = p[f"{row_key}_start_y"]
    count = int(p[f"{row_key}_count"])

    z = round(ph / 2, 10)

    for i in range(count):
        y = round(start_y + i * spacing, 10)
        idx = i - count // 2
        if idx == 0:
            suffix = "0"
        elif idx > 0:
            suffix = f"p{idx}"
        else:
            suffix = f"n{abs(idx)}"
        name = f"{row_prefix}_pillar_{suffix}"
        scene["cuboid"][name] = {
            "dims": [px, py, ph],
            "pose": [row_x, y, z, 1, 0, 0, 0],
        }


def generate(params: dict) -> dict:
    """根据参数生成场景 dict。"""
    p = PARAMS.copy()
    p.update({k: v for k, v in params.items() if v is not None})

    # 长条零件: 贴地, 沿 X 轴
    part_center_x = round(p["long_x_near"] + p["long_length"] / 2, 10)
    part_center_z = round(p["long_height"] / 2, 10)

    scene = {"cuboid": {}}

    scene["cuboid"]["long_part"] = {
        "dims": [p["long_length"], p["long_width"], p["long_height"]],
        "pose": [part_center_x, 0.0, part_center_z, 1, 0, 0, 0],
    }

    _add_row(scene, "row1", p, "row1")
    _add_row(scene, "row2", p, "row2")

    return scene


def main():
    parser = argparse.ArgumentParser(
        description="长条物件搬运避障场景生成器 — 汽车电池/轨道装备/铝型材搬运"
    )
    parser.add_argument("--output", "-o", default=None,
                        help="输出文件路径 (默认打印到 stdout)")
    for key, val in PARAMS.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", type=lambda x: x.lower() != 'false',
                                default=val, help=f"{key} (默认: {val})")
        else:
            parser.add_argument(f"--{key}", type=float, default=val,
                                help=f"{key} (默认: {val})")

    args = parser.parse_args()
    params = {k: getattr(args, k) for k in PARAMS}

    scene = generate(params)

    header = (
        "# myscene_1.yml — 长条形零件搬运避障场景\n"
        "# 场景: 汽车电池模组 / 轨道装备 / 铝型材搬运\n"
        "# 挑战: 双机械臂协同抬起长零件, 穿越高柱狭窄通道\n"
        "# 生成: python gen_scene_1.py\n"
        f"# 参数: {params}\n"
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
    else:
        print(header)
        print(yaml_str)


if __name__ == "__main__":
    main()
