# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""地形生成模块（Terrain）。

在训练启动时一次性生成一张「大地图高度场」，再切成
num_rows × num_cols 块子地形。每块对应一个课程格子：

  - 行 i（level）：同一类型内由易到难
  - 列 j（type）：地形种类（平地 / 坡 / 楼梯 / 障碍…）

生成结果供 LeggedRobot.create_sim() 使用：
  - heightsamples：测高 / 特权观测
  - env_origins[i,j]：机器人重置时放到哪一块
  - vertices/triangles：trimesh 模式下交给 PhysX

调用链（简述）::

    LeggedRobot.create_sim()
        → Terrain(cfg.terrain, num_envs)
            → 按 curriculum / selected / random 填高度场
            → 可选转成三角网格
"""

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class Terrain:
    """根据配置生成训练用地形网格。

    主要输出
    --------
    height_field_raw / heightsamples :
        整张地图的高度离散值（int16），单位是「垂直刻度」而非米。
    env_origins :
        形状 (num_rows, num_cols, 3)，每块子地形中心在世界系的放置点。
    vertices, triangles :
        仅 mesh_type=="trimesh" 时存在，用于 PhysX 三角网格。
    """

    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        """解析配置并生成整张地形。

        Args:
            cfg: 地形配置（LeggedRobotCfg.terrain 或其子类覆盖）
            num_robots: 并行环境数量（本文件主要用于兼容，布局不直接依赖它）
        """
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type

        # 平面 / 无地形：不生成高度场，环境里直接用平地
        if self.type in ["none", 'plane']:
            return

        # ---- 单块子地形的物理尺寸 [m] ----
        self.env_length = cfg.terrain_length  # 沿「行」方向长度
        self.env_width = cfg.terrain_width   # 沿「列」方向宽度

        # terrain_proportions 的前缀和，用于 make_terrain 里按 choice 选类型
        # 例：[0.1,0.2,0.3,0.3,0.1] → [0.1, 0.3, 0.6, 0.9, 1.0]
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        # dog 等任务可指定结构化课程类型列表，如 [0,1,2,...,8]
        self.curriculum_terrain_types = list(getattr(cfg, "curriculum_terrain_types", []))

        # 子地形总数 = 行数 × 列数
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        # 每块子地形的世界原点（之后机器人 reset 会读）
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        # ---- 像素尺度：米 → 高度场格子数 ----
        # horizontal_scale：每个格子代表多少米（如 0.1m）
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        # 大地图外围一圈边界（像素），防止机器人掉出地图
        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        # 整张高度场总尺寸 = 所有子块 + 两侧边界
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # 高度场原始缓冲（int16）；真实高度[m] = 值 * vertical_scale
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        # ---- 按配置选择生成策略 ----
        if cfg.curriculum:
            # difficulty >= 0：按比例混平地/复杂地（静态混合课程）
            # difficulty < 0（dog 默认）：结构化列课程（运行时再解锁列）
            if hasattr(cfg, "difficulty") and cfg.difficulty is not None and cfg.difficulty >= 0.0:
                self.difficulty_curriculum()
            else:
                self.curiculum()  # 注意：历史拼写保留为 curiculum
        elif cfg.selected:
            # 所有格子用同一种指定地形（调试用）
            self.selected_terrain()
        else:
            # 每个格子独立随机类型与难度
            self.randomized_terrain()

        # 供环境测高使用（与 height_field_raw 同一份数据）
        self.heightsamples = self.height_field_raw

        # trimesh：把高度场转成三角网格顶点/面片，交给 PhysX
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold,  # 过陡坡面校正为竖直面，避免网格病态
            )

    def randomized_terrain(self):
        """随机地形：每个子块独立抽类型与难度，再贴进大地图。"""
        for k in range(self.cfg.num_sub_terrains):
            # 把线性下标 k 还原成网格坐标 (行 i, 列 j)
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)  # 用于按 proportions 选地形种类
            difficulty = np.random.choice([0.5, 0.75, 0.9])  # 随机难度档
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def _adjust_step(self, step: float) -> float:
        """保证传给 Isaac Gym 地形工具的 step 不小于 vertical_scale。

        Isaac Gym 内部用 int(step / vertical_scale) 转成离散单位；
        若 step 太小会变成 0，后续建 range 时除零崩溃。
        """
        try:
            s = float(step)
        except Exception:
            return step
        return max(s, float(self.cfg.vertical_scale))

    def curiculum(self):
        """课程地形布局（函数名保留历史拼写 curiculum）。

        分支 A — 配置了 curriculum_terrain_types（dog 在用）::

            列 j → 课程阶段（类型 0~8）
            行 i → 该类型内部难度 0→1

        分支 B — 未配置结构化类型（旧版）::

            列 j → choice（间接决定 make_terrain 类型）
            行 i → difficulty
        """
        # ---- 分支 A：结构化 0~8 类型 ----
        if len(self.curriculum_terrain_types) > 0:
            for j in range(self.cfg.num_cols):
                stage_idx = self._curriculum_stage_index_from_col(j)
                terrain_type = self.curriculum_terrain_types[stage_idx]
                for i in range(self.cfg.num_rows):
                    # 第 0 行最易，最后一行最难
                    difficulty = i / max(self.cfg.num_rows - 1, 1)
                    terrain = self.make_structured_curriculum_terrain(terrain_type, difficulty)
                    self.add_terrain_to_map(terrain, i, j)
            return

        # ---- 分支 B：旧版「列选类型、行定难度」----
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001  # +ε 避免落在边界上歧义

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def difficulty_curriculum(self):
        """按 cfg.difficulty ∈ [0,1] 静态混合平地与复杂地形。

        - difficulty=0：几乎全平地
        - difficulty=1：复杂列占比达到 max_rough_terrain_ratio

        前 flat_cols 列全平；后面 rough_cols 列用 make_terrain，
        行越高（i 越大）复杂程度越高。
        """
        difficulty_level = float(np.clip(self.cfg.difficulty, 0.0, 1.0))
        max_rough_ratio = float(np.clip(getattr(self.cfg, "max_rough_terrain_ratio", 0.5), 0.0, 1.0))
        rough_ratio = difficulty_level * max_rough_ratio
        rough_cols = int(round(self.cfg.num_cols * rough_ratio))
        rough_cols = min(max(rough_cols, 0), self.cfg.num_cols)
        flat_cols = self.cfg.num_cols - rough_cols

        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                # 平地区：低难度训练时优先保证大量平地格子
                if j < flat_cols:
                    terrain = self.make_flat_terrain()
                else:
                    rough_col_idx = j - flat_cols
                    rough_col_count = max(rough_cols, 1)
                    # 在复杂列之间均匀分配不同 terrain choice
                    rough_choice = (rough_col_idx + 0.5) / rough_col_count
                    row_progress = (i + 1) / max(self.cfg.num_rows, 1)
                    rough_difficulty = np.clip(difficulty_level * row_progress, 0.0, 1.0)
                    terrain = self.make_terrain(rough_choice, rough_difficulty)

                self.add_terrain_to_map(terrain, i, j)

    def make_flat_terrain(self):
        """生成一块高度全 0 的平地子地形。"""
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        terrain.height_field_raw[:, :] = 0
        return terrain

    def _curriculum_stage_index_from_col(self, col):
        """把列索引映射到 curriculum_terrain_types 的阶段下标。

        保证从第 0 列到最后一列阶段单调不减。
        例如 num_cols=9、types 长度=9 → 列 j 直接对应阶段 j。
        """
        stage_count = len(self.curriculum_terrain_types)
        if stage_count == 0:
            return 0
        if self.cfg.num_cols <= 1:
            return 0
        # col/(num_cols-1) ∈ [0,1] → floor 到 [0, stage_count-1]
        ratio = col / (self.cfg.num_cols - 1)
        stage_idx = int(np.floor(ratio * stage_count))
        return min(max(stage_idx, 0), stage_count - 1)

    def make_structured_curriculum_terrain(self, terrain_type, difficulty):
        """按固定类型编号生成一块子地形（dog 结构化课程用）。

        类型编号
        --------
        0 平地
        1 平地 + 小起伏
        2 平地 + 大起伏
        3 斜坡
        4 斜坡 + 小起伏
        5 斜坡 + 大起伏
        6 下楼（台阶高度为负）
        7 上楼
        8 离散障碍方块

        Args:
            terrain_type: 0~8 的整数类型
            difficulty: [0,1]，同一类型内随行号增大而变难（起伏更大、坡更陡等）
        """
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        d = float(np.clip(difficulty, 0.0, 1.0))

        # 随难度线性放大的几何参数（单位：米或无量纲斜率）
        rough_light = 0.004 + 0.018 * d   # 轻起伏幅度
        rough_heavy = 0.012 + 0.040 * d   # 重起伏幅度
        slope = 0.05 + 0.35 * d           # 坡度
        stair_h = 0.03 + 0.12 * d         # 台阶高度
        obs_h = 0.03 + 0.15 * d           # 障碍高度

        if terrain_type == 0:
            # 纯平地
            terrain.height_field_raw[:, :] = 0
        elif terrain_type == 1:
            # 轻度随机起伏
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-rough_light,
                max_height=rough_light,
                step=self._adjust_step(0.0025),
                downsampled_scale=0.2,
            )
        elif terrain_type == 2:
            # 重度随机起伏
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-rough_heavy,
                max_height=rough_heavy,
                step=self._adjust_step(0.003),
                downsampled_scale=0.2,
            )
        elif terrain_type == 3:
            # 金字塔形斜坡（中间平台）
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
        elif terrain_type == 4:
            # 斜坡 + 轻起伏
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-rough_light,
                max_height=rough_light,
                step=self._adjust_step(0.0025),
                downsampled_scale=0.2,
            )
        elif terrain_type == 5:
            # 斜坡 + 重起伏
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-rough_heavy,
                max_height=rough_heavy,
                step=self._adjust_step(0.003),
                downsampled_scale=0.2,
            )
        elif terrain_type == 6:
            # 下楼：step_height 为负
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.30, step_height=-stair_h, platform_size=3.0
            )
        elif terrain_type == 7:
            # 上楼：step_height 为正
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.30, step_height=stair_h, platform_size=3.0
            )
        elif terrain_type == 8:
            # 随机矩形障碍块
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                obs_h,
                min_size=1.0,
                max_size=2.0,
                num_rects=20,
                platform_size=3.0,
            )
        else:
            # 未知类型兜底为平地
            terrain.height_field_raw[:, :] = 0

        return terrain

    def selected_terrain(self):
        """所有子块使用同一种配置指定的地形（调试/消融用）。

        要求 cfg.terrain_kwargs 含 'type' 字段，值为可 eval 的函数名字符串，
        其余参数传给该生成函数。
        """
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            # 动态调用如 pyramid_sloped_terrain(...)
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        """旧版通用地形工厂：用 choice 选种类，用 difficulty 定难度。

        choice 与 self.proportions（前缀和）比较，落入哪个区间就生成哪种地形：
          缓坡 / 崎岖坡 / 上下楼 / 离散障碍 / 踏石 / 沟 / 坑

        Args:
            choice: [0,1] 左右的浮点，决定地形种类
            difficulty: [0,1]，放大坡度、台阶、障碍等几何量
        """
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        # ---- 随难度缩放的几何参数 ----
        slope = difficulty * 0.4
        amplitude = 0.01 + 0.07 * difficulty
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)  # 越难石头越小
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty

        # ---- 按 proportions 区间选类型 ----
        if choice < self.proportions[0]:
            # 平滑斜坡；前半区间用负坡（下坡）
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            # 斜坡 + 随机起伏
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-amplitude,
                max_height=amplitude,
                step=self._adjust_step(0.005),
                downsampled_scale=0.2,
            )
        elif choice < self.proportions[3]:
            # 楼梯；proportions[2] 前为下楼（负高度）
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.30, step_height=step_height, platform_size=3.
            )
        elif choice < self.proportions[4]:
            # 离散矩形障碍
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.,
            )
        elif choice < self.proportions[5]:
            # 踏石路
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.,
                platform_size=4.,
            )
        elif choice < self.proportions[6]:
            # 沟壑
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            # 深坑
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        """把一块子地形贴到大高度场的 (row, col) 位置，并记录原点。

        坐标系
        ------
        大地图外围有 border 像素边框；
        第 (i,j) 块占用::

            x ∈ [border + i*L, border + (i+1)*L)
            y ∈ [border + j*W, border + (j+1)*W)

        原点取块中心，z 取中心附近最高点（避免出生点陷进坑里）。

        Args:
            terrain: Isaac Gym SubTerrain，含 height_field_raw
            row: 行索引 i（难度 level）
            col: 列索引 j（类型 type）
        """
        i = row
        j = col
        # ---- 写入大高度场对应切片 ----
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        # ---- 计算该块世界原点 [m] ----
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        # 中心 ±1m 窗口内的最高高度 → 出生高度
        x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.):
    """在子地形中心挖一圈沟，中间保留平台（跨越训练用）。

    实现：先把更大区域高度置为极低（-1000），再把中心平台填回 0。
    """
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    # 外圈深沟
    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
    # 中心平台恢复平地
    terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.):
    """在子地形中心挖一个方形深坑。"""
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
