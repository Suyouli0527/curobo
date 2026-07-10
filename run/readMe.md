# 双臂 Panda 避障可视化

cuRobo reactive MPC + Viser，支持双臂协同运动与避障规划。
陈思丞

## 启动

```bash
cd /home/sicheng/csc_curobo
source .venv/bin/activate

# 默认场景 (myscene_1.yml)
python -m run.main

# 指定场景
python -m run.main --scene myscene_2.yml

# 指定端口
python -m run.main --port 8084

# 指定机器人
python -m run.main --robot dual_panda.yml --scene my_scene.yml
```

浏览器访问 `http://localhost:8080`。

## 操作

- **拖拽绿色控制框** → 机器人实时追踪
- **点击 Coordinate** → 双臂协同模式（橙色），拖任一侧另一侧自动跟随
- **再次点击 Coordinate** → 退出协同（灰色），双臂独立

## 目录结构

```
run/
  main.py            入口脚本
  controller.py      控制层 — DualArmMPCController (MPC, 约束, 协同同步)
  scene.py           可视化层 — Viser, 障碍物渲染, 控制框管理
  utils.py           数学工具 — 四元数 (torch/numpy), 位姿转换
  myScene/           场景文件
    my_scene.yml      基础测试场景
    myscene_1.yml     长条物体搬运避障
    myscene_2.yml     多层货架拣选 (长条物体 + 支架)
    myscene_3.yml     微型椅子 + 高柱障碍
    myscene_5.yml     椅子装配 + 多障碍柱
  readMe.md
```

## 自定义场景

在 `myScene/` 下新建 `.yml` 文件：

```yaml
cuboid:
  obstacle_name:
    dims: [dx, dy, dz]          # 尺寸 (米)
    pose: [x, y, z, qw, qx, qy, qz]  # 位姿
```

加载优先级：`myScene/` > `curobo内置` > 根目录。

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--robot` | `dual_franka_with_camera.yml` | 机器人配置文件 |
| `--scene` | `myscene_1.yml` | 场景文件名 |
| `--port` | `8080` | Viser 网页端口 |
