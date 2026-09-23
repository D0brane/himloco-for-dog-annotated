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

"""腿式 / 轮腿机器人 Isaac Gym 环境实现（LeggedRobot）。

本文件是 legged_gym 的核心环境类，被 Dog / Go2w 等薄封装直接继承。
配置默认值见同目录 legged_robot_config.py；具体机器人在 envs/dog、envs/go2w 中覆盖。

主要职责
--------
1. 创建 PhysX 仿真、地形与并行环境（create_sim / _create_envs）
2. 策略步进：动作 → 力矩 → 多子步仿真 → 奖励 / 终止 / 观测（step）
3. 拼装 Actor 观测与 Critic 特权观测（compute_observations）
4. 按配置动态挂接奖励项（_prepare_reward_function + _reward_*）
5. 域随机化、地形/指令课程、关节与轮索引管理

观测布局（单步策略可见部分，前 num_one_step_obs 维）
------------------------------------------------
  ang_vel(3) | projected_gravity(3) | commands(3)
  | dof_err(n) | dof_vel(n) | last_actions(n)
其后还可拼接特权量：base_lin_vel(3)、disturbance(3)、高度扫描、足端接触力等。
Actor 只用前 num_one_step_obs（并堆叠 history）；Critic 用特权维。

控制约定（轮腿）
----------------
  腿关节：位置 PD，目标 = default + action_scale * action
  轮关节：速度目标 vel_ref = wheel_direction * action * vel_scale（位置误差强制为 0）
"""

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    """腿式机器人并行仿真环境，继承 BaseTask。

    继承关系::

        BaseTask          # Gym 句柄、obs/rew 缓冲、viewer
            └── LeggedRobot   # 本类：机器人逻辑
                    └── Dog / Go2w  # 通常只绑定 cfg，不重写逻辑

    与 RL 训练的接口（供 HIMOnPolicyRunner 调用）::

        obs, priv, rew, reset, extras, term_ids, term_priv = env.step(actions)
    """
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """解析配置 → 创建仿真/地形/环境 → 初始化训练用缓冲区与奖励表。

        Args:
            cfg: 环境配置（LeggedRobotCfg 或其子类，如 DogRoughCfg）
            sim_params: Isaac Gym SimParams（步长、重力、PhysX 等）
            physics_engine: 物理引擎类型，本项目使用 PhysX
            sim_device: 如 'cuda:0' 或 'cpu'
            headless: True 则不创建渲染窗口
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None  # 地形高度采样网格（heightfield/trimesh 时填充）
        self.debug_viz = False  # True 时绘制高度测量点（很慢）
        self.init_done = False  # 初始化完成前，地形课程不更新
        self._parse_cfg(self.cfg)  # 派生 dt、reward_scales、episode 长度等
        # BaseTask.__init__ 内会调用本类的 create_sim()
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations  # Actor 单步观测维
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs  # Critic 单步维
        self.history_length = int(self.num_obs / self.num_one_step_obs)  # 历史帧数，如 6

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()  # 包装 GPU 状态张量、PD 增益、域随机化缓冲等
        self._prepare_reward_function()  # 按 scales 挂接 _reward_* 并乘 dt
        self.init_done = True

    def step(self, actions):
        """执行一个策略步：裁剪动作 →（可选）延迟 → 多子步仿真 → 后处理。

        Args:
            actions: (num_envs, num_actions) 网络输出的原始动作

        Returns:
            obs_buf: Actor 观测（含历史堆叠）
            privileged_obs_buf: Critic 特权观测
            rew_buf: 本步奖励
            reset_buf: 是否需要重置（摔倒或超时）
            extras: 回合统计、timeout 等
            termination_ids: 本步终止的环境 ID
            termination_priveleged_obs: 终止瞬间的特权观测（供价值估计 bootstrap）
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # 将动作展开到每个仿真子步；若开启 delay，在子步间从 last_actions 过渡到 actions
        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
        self.render()
        # 一个策略步 = decimation 个仿真步（策略频率 = 1 / (decimation * sim.dt)）
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # 裁剪观测后返回给算法
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        """物理子步全部结束后的统一处理。

        顺序：刷新状态 → 机体/足端量 → 回调（指令/推扰/测高）
        → 终止判定 → 奖励 → 重置摔倒环境 → 拼观测 → 滚动历史。
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 世界系速度转到机体系；投影重力用于姿态观测
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # rigid_body_state 每刚体 13 维：pos(3)+quat(4)+lin_vel(3)+ang_vel(3)
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self._post_physics_step_callback()  # 重采样指令、测高、推机器人、外力扰动

        # 注意：奖励在 reset 之前算，用的是本步终止前的状态
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations()  # 重置后再拼观测，供下一步策略使用

        # 滚动动作/速度历史，供 action_rate、平滑项与下一步延迟使用
        self.disturbance[:, :, :] = 0.0
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """判定哪些环境需要重置。

        - 指定连杆（如 base_link）接触力超阈值 → 失败终止
        - 回合时长超过 max_episode_length → 超时（通常不加 termination 惩罚）
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # 超时，不给予终止奖励
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """重置指定环境：课程更新 → 状态/指令 → 缓冲 → 域随机化 → 写 extras。

        Args:
            env_ids: 需要重置的环境 ID（可为空）
        """
        if len(env_ids) == 0:
            return
        # 先按本回合表现更新地形课程，再重置位姿
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # 指令课程对全体环境共享范围，只在「整回合长度」边界上更新一次
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # 重置机器人状态
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # 清空与历史相关的缓冲，避免跨回合串扰
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        # 更新高度测量
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # 重置时重新采样部分域随机化参数
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), 1), device=self.device)
        self.refresh_actor_rigid_shape_props(env_ids)

        # 回合平均奖励写入 extras，供 TensorBoard 记录
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        # 记录课程额外信息
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            self.extras["episode"]["terrain_type_col_max"] = float(self.curriculum_max_type_col)
            self.extras["episode"]["terrain_success_ratio"] = float(self.last_terrain_success_ratio)
            self.extras["episode"]["terrain_timeout_ratio"] = float(self.last_terrain_timeout_ratio)
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # 向算法发送超时信息（超时通常不当作 bootstrap 的「真终止」）
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0

    def compute_reward(self):
        """累加所有已启用奖励项。

        每项：rew = _reward_<name>() * reward_scales[name]（scale 已含 dt）。
        only_positive_rewards 时先把总和截到 ≥0，再可选加 termination。
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # termination 在裁剪之后单独加，避免被 only_positive 抹掉负惩罚
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """拼装当前观测并写入历史队列（FIFO）。

        完整 current_obs 拼接顺序::

            [策略可见 57/45 维]
              ang_vel(3), gravity(3), cmd(3), dof_err(n), dof_vel(n), actions(n)
            [特权追加]
              base_lin_vel(3), disturbance(3),
              heights(≈187, 可选), feet_contact_forces(3*num_feet)

        - obs_buf 只取前 num_one_step_obs，再与旧历史拼接
        - privileged_obs_buf 取前 num_one_step_privileged_obs
        - 轮关节位置/误差在观测中强制为 0（连续旋转无绝对角意义）
        """
        self.dof_err = self.dof_pos - self.default_dof_pos
        self.dof_err[:,self.wheel_indices] = 0   # 车轮关节误差置零
        self.dof_pos[:,self.wheel_indices] = 0   # 车轮关节位置置零（不参与观测）

        # ---- 策略可见部分 ----
        current_obs = torch.cat((   self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    self.dof_err* self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # 仅对策略可见段加噪声（指令与动作通常不加）
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # ---- 特权信息（仿真可得，真机通常没有或很贵）----
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        # 足端三维接触力（归一化后）
        contact_forces_scale, contact_forces_shift = get_scale_shift(self.cfg.normalization.contact_force_range)
        contact_forces = (self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1)
                  - contact_forces_shift) * contact_forces_scale
        current_obs = torch.cat((current_obs, contact_forces), dim=-1)

        # FIFO：最新一帧放在最前，丢掉最旧一帧
        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)

    def get_current_obs(self):
        """ 获取当前时刻观测（不含历史），用于调试或特权观测生成
        """
        self.dof_err = self.dof_pos - self.default_dof_pos 
        self.dof_err[:,self.wheel_indices] = 0 
        self.dof_pos[:,self.wheel_indices] = 0 

        current_obs = torch.cat((   self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    self.dof_err* self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        contact_forces_scale, contact_forces_shift = get_scale_shift(self.cfg.normalization.contact_force_range)
        contact_forces = (self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1)
                  - contact_forces_shift) * contact_forces_scale
        current_obs = torch.cat((current_obs, contact_forces), dim=-1)

        return current_obs
        
    def compute_termination_observations(self, env_ids):
        """ 计算终止时的观测（用于给学习算法提供终止时刻的观测）
        """
        self.dof_err = self.dof_pos - self.default_dof_pos 
        self.dof_err[:,self.wheel_indices] = 0 
        self.dof_pos[:,self.wheel_indices] = 0 

        current_obs = torch.cat((   self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    self.dof_err* self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)
        
        contact_forces_scale, contact_forces_shift = get_scale_shift(self.cfg.normalization.contact_force_range)
        contact_forces = (self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1)
                  - contact_forces_shift) * contact_forces_scale
        current_obs = torch.cat((current_obs, contact_forces), dim=-1)

        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]
        
            
    def create_sim(self):
        """ 创建仿真、地形和环境
        """
        self.up_axis_idx = 2   # 2表示Z轴向上，1表示Y轴向上，需相应调整重力
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ 设置相机位置和朝向
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- 回调函数 --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ 回调：允许存储/更改/随机化每个环境的刚体形状属性。
            在环境创建期间调用。
            基础行为：随机化每个环境的摩擦系数。

        Args:
            props (List[gymapi.RigidShapeProperties]): 资产中每个形状的属性列表
            env_id (int): 环境ID

        Returns:
            [List[gymapi.RigidShapeProperties]]: 修改后的刚体形状属性
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # 准备摩擦随机化
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # 准备恢复系数随机化
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        """ 刷新指定环境的刚体形状属性（用于重置时重新随机化）
        """
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ 回调：允许存储/更改/随机化每个环境的DOF属性。
            在环境创建期间调用。
            基础行为：存储URDF中定义的位置、速度和力矩限制。

        Args:
            props (numpy.array): 资产中每个DOF的属性数组
            env_id (int): 环境ID

        Returns:
            [numpy.array]: 修改后的DOF属性
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                # 从URDF读取力矩限制，但允许通过配置文件缩放，便于调参而不修改URDF文件
                base_effort = props["effort"][i].item()
                scale = getattr(self.cfg.control, "torque_limit_scale", 1.0)
                self.torque_limits[i] = base_effort * float(scale)
                # 软限制：为奖励函数设置关节位置的软限制边界
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        """ 回调：允许修改每个环境的刚体属性（如质量、质心）
        """
        # 随机化基座质量
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            
        # 随机化质心偏移
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        # 随机化连杆质量
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ 在计算终止、奖励和观测之前调用的回调。
            默认行为：根据目标和航向计算角速度命令，测量地形高度，随机推动机器人。
        """
        # 重新采样命令
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -2., 2.)

        # 测量高度
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        # 随机推动机器人
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        # 随机添加外力扰动
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _resample_commands(self, env_ids):
        """ 为某些环境随机选择新命令

        Args:
            env_ids (List[int]): 需要新命令的环境ID列表
        """
        # 使用可配置的x速度范围，而非硬编码的[-1,1]
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # 高速度环境（前20%环境）的x速度范围更宽
        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]

        self.commands[high_vel_env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(high_vel_env_ids), 1), device=self.device).squeeze(1)

        # 将高速度环境的y命令置零（如果速度不够高则保留）
        self.commands[high_vel_env_ids, 1:2] *= (torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1) < 1.0).unsqueeze(1)

        # 将小命令置零
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """根据策略动作计算发送给仿真的关节力矩。

        轮腿混合控制（control_type == "P" 时）::

            腿关节:
              q_des = default + action_scale * action
              τ = Kp*(q_des - q) + Kd*(0 - dq)   （腿的 vel_ref 为 0）
            轮关节:
              不做位置跟踪（dof_err/actions_scaled 对应分量置 0）
              ω_des = wheel_direction * action * vel_scale
              τ = Kd*(ω_des - dq)                （轮的 Kp 通常为 0）

        力矩维数必须等于 DOF 数；最后按 URDF effort（可再乘 torque_limit_scale）裁剪。

        Args:
            actions: (num_envs, num_actions) 当前子步使用的动作

        Returns:
            裁剪后的力矩张量
        """
        # 位置误差：default - q；轮关节不做位置伺服
        dof_err = self.default_dof_pos - self.dof_pos
        dof_err[:,self.wheel_indices] =  0  # 车轮关节不进行位置误差控制
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.wheel_indices] = 0  # 轮不走「位置偏移」通道
        vel_ref = torch.zeros_like(actions_scaled)
        wheel_direction = getattr(self.cfg.control, "wheel_direction", 1.0)
        vel_tmp = actions * self.cfg.control.vel_scale
        # 仅轮关节写入速度目标；方向由 wheel_direction 约定（+1/-1）
        vel_ref[:, self.wheel_indices] = wheel_direction * vel_tmp[:, self.wheel_indices]
        control_type = self.cfg.control.control_type

        if control_type=="P":  # 位置(+轮速度) PD
            # τ = Kp*Kp_rand*(Δq_action + (q0-q)) + Kd*Kd_rand*(ω_ref - dq)
            torques = self.p_gains * self.Kp_factors * (actions_scaled + dof_err) + self.d_gains * self.Kd_factors * (vel_ref - self.dof_vel)
        elif control_type=="V":  # 纯速度控制
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":  # 动作直接当力矩
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ 重置指定环境的DOF位置和速度。
            位置在默认位置的0.5~1.5倍范围内随机选择，速度设为零。

        Args:
            env_ids (List[int]): 环境ID列表
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ 重置指定环境根状态（位置和速度）。
            根据课程设置基座位置，随机选择基座速度在[-0.5,0.5]范围内。

        Args:
            env_ids (List[int]): 环境ID列表
        """
        # 基座位置
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy位置在中心1m内
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # 基座速度
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]:线速度，[10:13]:角速度
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ 随机推动机器人。通过设置随机基座速度模拟脉冲。
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # 线速度x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_robots(self):
        """ 随机对机器人施加外力扰动。
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _update_terrain_curriculum(self, env_ids):
        """地形课程：行=难度 level，列=地形类型 type。

        单环境：
          - 走得够远且线速度跟踪达标 → level +1
          - 走得太近 → level -1
        全局：
          - 达标率 + 存活率够高 → 解锁下一列类型（正常升课）
          - 卡住太久但尚有最低表现 → 保底解锁
        重置时在已解锁列范围内重采样 terrain_types。
        """
        # 实施地形课程
        if not self.init_done:
            # 初始重置时不改变课程
            return

        # 课程晋级门槛：线速度跟踪达到阈值才允许升级地形难度。
        tracking_rew = self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length
        pass_threshold = self.cfg.terrain.command_tracking_pass_threshold * self.reward_scales["tracking_lin_vel"]
        tracking_pass = tracking_rew > pass_threshold

        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # 行走足够远的机器人晋级到更难的地形
        move_up = (distance > self.terrain.env_length / 2) & tracking_pass
        # 行走距离不足所需距离一半的机器人降级到更简单的地形
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

        # 通过率与存活率共同达标后逐步放开更多地形列；若长期卡住则触发保底解锁。
        success_ratio = torch.mean(tracking_pass.float()) if len(env_ids) > 0 else torch.tensor(0.0, device=self.device)
        self.last_terrain_success_ratio = float(success_ratio.item())
        timeout_ratio = torch.mean(self.time_out_buf[env_ids].float()) if len(env_ids) > 0 else torch.tensor(0.0, device=self.device)
        self.last_terrain_timeout_ratio = float(timeout_ratio.item())
        unlock_interval_steps = max(1, int(self.cfg.terrain.curriculum_unlock_interval_episodes * self.max_episode_length))
        force_unlock_interval_steps = max(1, int(self.cfg.terrain.curriculum_force_unlock_interval_episodes * self.max_episode_length))
        steps_since_unlock = self.common_step_counter - self.last_terrain_unlock_step
        can_normal_unlock = (success_ratio > self.cfg.terrain.command_tracking_success_ratio) and (timeout_ratio > self.cfg.terrain.curriculum_min_timeout_ratio) and (steps_since_unlock >= unlock_interval_steps)
        can_force_unlock = (steps_since_unlock >= force_unlock_interval_steps) and (success_ratio > self.cfg.terrain.curriculum_force_unlock_min_success_ratio) and (timeout_ratio > self.cfg.terrain.curriculum_force_unlock_min_timeout_ratio)
        if (can_normal_unlock or can_force_unlock) and self.curriculum_max_type_col < (self.cfg.terrain.num_cols - 1):
            self.curriculum_max_type_col = min(
                self.curriculum_max_type_col + self.cfg.terrain.curriculum_type_unlock_step,
                self.cfg.terrain.num_cols - 1,
            )
            self.last_terrain_unlock_step = self.common_step_counter

        # 每次 reset 时在已解锁范围内重采样地形类型，实现“部分环境进入新课程”。
        self.terrain_types[env_ids] = torch.randint(
            0,
            self.curriculum_max_type_col + 1,
            (len(env_ids),),
            device=self.device,
            dtype=torch.long,
        )

        # 解决最后一关的机器人被发送到随机关卡
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (最小等级为零)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """指令课程：高低速两组环境都跟踪够好时，扩大 lin_vel_x 采样范围。

        前 20% 环境视为「高速组」，其余为「低速组」；两侧都过门槛才扩范围，
        上限/下限受 max_curriculum 约束。
        """
        low_vel_env_ids = (env_ids > (self.num_envs * 0.2))
        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        if len(low_vel_env_ids) == 0 or len(high_vel_env_ids) == 0:
            return
        # 如果跟踪奖励超过最大值的80%，则增加命令范围
        pass_threshold = self.cfg.commands.curriculum_pass_threshold * self.reward_scales["tracking_lin_vel"]
        step = self.cfg.commands.curriculum_step
        if (torch.mean(self.episode_sums["tracking_lin_vel"][low_vel_env_ids]) / self.max_episode_length > pass_threshold) and (torch.mean(self.episode_sums["tracking_lin_vel"][high_vel_env_ids]) / self.max_episode_length > pass_threshold):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - step, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + step, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ 设置用于缩放观测噪声的向量。
            [注]：当观测结构变化时必须适应此方法。

        Args:
            cfg (Dict): 环境配置文件

        Returns:
            [torch.Tensor]: 用于缩放[-1,1]均匀分布的尺度向量
        """
        # 根据观测结构构建噪声尺度向量
        if self.cfg.terrain.measure_heights:
            noise_vec = torch.zeros(9 + 3*self.num_actions + 187, device=self.device)
        else:
            noise_vec = torch.zeros(9 + 3*self.num_actions, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # projected_gravity (3)
        noise_vec[3:6] = noise_scales.gravity * noise_level
        # commands (3)
        noise_vec[6:9] = 0.
        # dof_err (num_actions)
        noise_vec[9:(9 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # dof_vel (num_actions)
        noise_vec[(9 + self.num_actions):(9 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # actions (num_actions)
        noise_vec[(9 + 2 * self.num_actions):(9 + 3 * self.num_actions)] = 0.
        # height_measurements (187, if enabled)
        if self.cfg.terrain.measure_heights:
            noise_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 187)] = \
                noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ 初始化包含仿真状态和处理后量的PyTorch张量
        """
        # 获取gym GPU状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 创建包装张量以便切片
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # 形状: num_envs, num_bodies, xyz轴

        # 初始化后续使用的数据
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.diagonal_leg_indices = self._infer_diagonal_leg_indices()
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()
        self.base_height_points = self._init_base_height_points()

        # 关节位置偏移和PD增益
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            print(i)
            print(name)
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # 识别髋关节索引
        hip_ids = []
        for i, name in enumerate(self.dof_names):
            is_hip = name.endswith("hip_joint")
            is_leg_hip = name.endswith("leg_joint") and ("bleg" not in name) and ("sleg" not in name)
            if is_hip or is_leg_hip:
                hip_ids.append(i)
        self.hip_indices = torch.tensor(hip_ids, dtype=torch.long, device=self.device, requires_grad=False)

        # 随机化kp, kd, 电机强度等
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        # 存储摩擦和恢复系数
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)


    def _prepare_reward_function(self):
        """根据 cfg.rewards.scales 动态挂接奖励函数。

        规则：
          - scale == 0 → 丢弃该项（不调用对应 _reward_*）
          - scale != 0 → scale *= dt（使回报对策略步长大致不变）
          - 方法名约定：配置键 ``tracking_lin_vel`` ↔ ``_reward_tracking_lin_vel``
          - ``termination`` 单独处理，不进 reward_functions 列表
        """
        # 移除零尺度，并将非零尺度乘以dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # 准备函数列表
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # 每个启用项的回合累计，用于日志 / 课程判定
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}


    def _create_ground_plane(self):
        """ 向仿真添加地平面，根据配置设置摩擦和恢复系数。
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ 向仿真添加高度场地形，根据配置设置参数。
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ 向仿真添加三角形网格地形，根据配置设置参数。
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ 创建环境：
             1. 加载机器人URDF/MJCF资产，
             2. 对每个环境：
                2.1 创建环境，
                2.2 调用DOF和刚体形状属性回调，
                2.3 使用这些属性创建actor并添加到环境中
             3. 存储机器人不同部分的索引
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # 保存资产中的身体名称
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        self.feet_names = feet_names
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        
        # 车轮关节名称
        wheel_names =[]
        for name in self.cfg.asset.wheel_name:
            wheel_names.extend([s for s in self.dof_names if name in s])
        print("###self.rigid_body names:",body_names)
        print("###self.dof names:",self.dof_names)
        print("###penalized_contact_names:",penalized_contact_names)
        print("###termination_contact_names:",termination_contact_names)
        print("###feet_names:",feet_names)
        print("###wheels name:",wheel_names)
        
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        for i in range(self.num_envs):
            # 创建环境实例
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # ---- 刚体 / DOF 索引表（后续奖励、观测、控制都靠它们切片）----
        # feet_indices: 足端刚体 → 接触力、足端位姿
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        # penalised_contact_indices: collision 惩罚用连杆
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        # termination_contact_indices: 触地即失败（如 base_link）
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        # wheel_indices: 轮式 DOF（部署端常对应 [3,7,11,15]）
        self.wheel_indices = torch.zeros(len(wheel_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wheel_names)):
            self.wheel_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], wheel_names[i])

        # leg_indices: 非轮 DOF，用于 leg_motion 等「少动腿、多用轮」奖励
        all_dof_indices = torch.arange(self.num_dof, dtype=torch.long, device=self.device, requires_grad=False)
        if self.wheel_indices.numel() > 0:
            wheel_mask = torch.zeros(self.num_dof, dtype=torch.bool, device=self.device, requires_grad=False)
            wheel_mask[self.wheel_indices] = True
            self.leg_indices = all_dof_indices[~wheel_mask]

            # 确保车轮关节使用力矩控制模式（驱动模式为力矩）。
            # 某些资产或导入器可能覆盖默认驱动模式；强制将车轮DOF设置为力矩控制，
            # 以便 set_dof_actuation_force_tensor 能正确地将计算出的力矩应用到车轮上。
            try:
                wheel_idx_list = self.wheel_indices.cpu().numpy().tolist()
                for env_handle, actor_handle in zip(self.envs, self.actor_handles):
                    dp = self.gym.get_actor_dof_properties(env_handle, actor_handle)
                    for wi in wheel_idx_list:
                        # 设置驱动模式为力矩控制
                        dp['driveMode'][wi] = int(gymapi.DOF_MODE_EFFORT)
                        # 将刚度和阻尼设为零，以实现纯力矩控制
                        dp['stiffness'][wi] = 0.0
                        dp['damping'][wi] = 0.0
                    self.gym.set_actor_dof_properties(env_handle, actor_handle, dp)
            except Exception:
                # 调试辅助函数出错时不中断创建过程
                pass
        else:
            self.leg_indices = all_dof_indices


    def _get_env_origins(self):
        """ 设置环境原点。在粗糙地形上，原点由地形平台定义，否则创建网格。
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # 将机器人放置在地形定义的原点
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            # 从最容易的地形列起步，后续由课程逻辑逐步解锁更多列。
            self.curriculum_max_type_col = int(np.clip(getattr(self.cfg.terrain, "curriculum_start_type_col", 0), 0, self.cfg.terrain.num_cols - 1))
            self.last_terrain_unlock_step = -int(self.cfg.terrain.curriculum_unlock_interval_episodes * self.max_episode_length)
            self.last_terrain_success_ratio = 0.0
            self.last_terrain_timeout_ratio = 0.0
            self.terrain_types = torch.randint(
                0,
                self.curriculum_max_type_col + 1,
                (self.num_envs,),
                device=self.device,
                dtype=torch.long,
            )
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # 创建机器人网格
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        """从 cfg 派生运行时常量（在创建仿真前调用）。

        - dt: 策略步长 = decimation * sim.dt
        - reward_scales / command_ranges: 嵌套类转 dict，便于热改与课程
        - 非 heightfield/trimesh 时强制关闭地形课程
        - push_interval: 秒 → 策略步数
        """
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ 绘制调试可视化（会显著减慢仿真速度）。
            默认行为：绘制高度测量点。
        """
        # 绘制高度线
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ 返回测量高度的点（在基座坐标系中）

        Returns:
            [torch.Tensor]: 形状为 (num_envs, self.num_height_points, 3) 的张量
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_base_height_points(self):
        """ 返回基座周围测量高度的点（用于计算基座高度偏移）

        Returns:
            [torch.Tensor]: 形状为 (num_envs, self.num_base_height_points, 3) 的张量
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ 在机器人周围的指定点采样地形高度。
            这些点通过基座位置偏移并绕基座偏航旋转。

        Args:
            env_ids (List[int], optional): 需要返回高度的环境子集，默认为None。

        Returns:
            [torch.Tensor]: 高度值，形状 (num_envs, num_height_points)
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _get_base_heights(self, env_ids=None):
        """ 采样机器人基座周围点的高度，计算基座相对于地形的平均高度（用于奖励）。

        Args:
            env_ids (List[int], optional): 环境子集，默认为None。

        Returns:
            [torch.Tensor]: 基座相对高度，形状 (num_envs,)
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height
    
    def _get_feet_heights(self, env_ids=None):
        """ 采样脚部接触点地形高度，计算脚部相对于地形的高度。

        Args:
            env_ids (List[int], optional): 环境子集，默认为None。

        Returns:
            [torch.Tensor]: 脚部相对高度，形状 (num_envs, num_feet)
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self.feet_pos[env_ids].clone()
        else:
            points = self.feet_pos.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        feet_height =  self.feet_pos[:, :, 2] - heights

        return feet_height

    # =========================================================================
    # 奖励函数（命名约定：_reward_<cfg.rewards.scales 中的键名>）
    # 返回值均为 (num_envs,)；正数表示「越大越好」的原始量，
    # 最终贡献 = 返回值 * scale（scale 已在 _prepare_reward_function 中乘过 dt）。
    # 惩罚项通常在 scales 里设为负数。
    # =========================================================================
    def _reward_tracking_lin_vel(self):
        """线速度跟踪：exp(-||v_cmd_xy - v_base_xy||^2 / σ)，完美跟踪≈1。"""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_wheel_speed(self):
        """有平移指令时鼓励轮转：mean(|ω_wheel|) * 1{|cmd_xy|>0.1}。"""
        if not hasattr(self, "wheel_indices") or self.wheel_indices.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        wheel_speed = torch.mean(torch.abs(self.dof_vel[:, self.wheel_indices]), dim=1)
        return wheel_speed * (cmd_mag > 0.1)

    def _reward_wheel_still(self):
        """近零平移指令时惩罚轮空转（与 wheel_speed 互补）。"""
        if not hasattr(self, "wheel_indices") or self.wheel_indices.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        wheel_speed = torch.mean(torch.abs(self.dof_vel[:, self.wheel_indices]), dim=1)
        return wheel_speed * (cmd_mag < 0.1)

    def _reward_leg_motion(self):
        """有平移指令时惩罚腿关节速度，促使「滚轮前进、少迈步」。"""
        if not hasattr(self, "leg_indices") or self.leg_indices.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        leg_speed = torch.mean(torch.abs(self.dof_vel[:, self.leg_indices]), dim=1)
        return leg_speed * (cmd_mag > 0.1)

    def _reward_tracking_ang_vel(self):
        """偏航角速度跟踪：exp(-(ωz_cmd - ωz)^2 / σ)。"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_lin_vel_z(self):
        """惩罚竖直方向速度（抑制蹦跳）。"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """惩罚滚转/俯仰角速度（抑制晃动）。"""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """惩罚机身非水平：投影重力的 xy 分量平方和。"""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """惩罚机身相对地形平均高度偏离 base_height_target。"""
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_foot_clearance(self):
        """惩罚足端离地高度偏离目标，并按足端水平速度加权（动得快时更在意离地）。"""
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])

        target = getattr(self.cfg.rewards, "clearance_height_target", 0.08)
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - target).view(self.num_envs, -1)
        foot_lateral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_lateral_vel, dim=1)

    def _reward_hip_default(self):
        """惩罚髋关节偏离默认角；硬编码索引 [0,4,8,12]（四条腿第一关节）。"""
        hip_err = torch.sum((self.dof_pos[:, [0, 4, 8, 12]] - self.default_dof_pos[:, [0, 4, 8, 12]]) ** 2, dim = 1)
        return hip_err

    def _reward_hip_pos(self):
        """同上，但使用 _init_buffers 里按名字识别的 hip_indices。"""
        if not hasattr(self, "hip_indices") or self.hip_indices.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        hip_pos_error = self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]
        return torch.sum(torch.square(hip_pos_error), dim=1)

    def _reward_stand_still(self):
        """近零指令时惩罚腿关节偏离默认姿态（轮关节不计）。"""
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:,self.wheel_indices] = 0
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_torques(self):
        """惩罚力矩平方和（能耗/冲击近似）。"""
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_collision(self):
        """惩罚 penalize_contacts_on 连杆上出现明显接触力的数量。"""
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_feet_stumble(self):
        """惩罚足端侧向力远大于法向力（撞竖直面/绊倒）。"""
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             3.0 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stumble(self):
        """feet_stumble 的别名，兼容旧配置键名。"""
        return self._reward_feet_stumble()

    def _reward_action_rate(self):
        """惩罚相邻步动作差（一阶平滑）。"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_smoothness(self):
        """惩罚动作二阶差分（更强平滑）。"""
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)

    def _reward_joint_power(self):
        """惩罚 |q̇|·|τ|（机械功率近似）。"""
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_dof_vel(self):
        """惩罚关节速度平方和；轮速度排除，避免与 wheel_speed 目标冲突。"""
        dof_vel = self.dof_vel
        if hasattr(self, "wheel_indices") and self.wheel_indices.numel() > 0:
            # 避免原地修改仿真状态缓冲区
            dof_vel = self.dof_vel.clone()
            dof_vel[:, self.wheel_indices] = 0
        return torch.sum(torch.square(dof_vel), dim=1)

    def _reward_dof_acc(self):
        """惩罚关节加速度 (Δq̇ / dt)^2。"""
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_pos_limits(self):
        """惩罚关节角越出软限位的部分。"""
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # 下极限
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        """惩罚关节速度越出软限位；单关节超量最多按 1 rad/s 计。"""
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        """惩罚力矩越出软限位的部分。"""
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_termination(self):
        """失败终止（非超时）时为 1，配合负 scale 作为摔倒惩罚。"""
        return self.reset_buf * ~self.time_out_buf

    def _reward_feet_contact_forces(self):
        """惩罚足端接触力超过 max_contact_force 的超量部分。"""
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_run_still(self):
        """有运动指令时惩罚腿仍停在默认姿态（避免「指令来了却冻住」）。"""
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:,self.wheel_indices] = 0
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)