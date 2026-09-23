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

"""腿式机器人默认配置（环境 + PPO）。

本文件提供「通用四足/轮腿」训练的默认超参模板。
具体机器人（如 dog / go2w）应继承本类并在各自 `*_config.py` 中覆盖：
  - 观测/动作维度
  - URDF、默认关节角、PD 增益
  - 奖励权重、地形课程等

配置通过 BaseConfig 递归实例化：写 `class env:` 后可用 `cfg.env.num_envs`。
"""

from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    """环境侧配置：仿真、观测、控制、奖励、域随机化等。

    与算法无关；算法相关参数见下方 LeggedRobotCfgPPO。
    """

    class env:
        """并行环境与观测/动作维度定义。"""
        num_envs = 4096  # 并行仿真环境数量（越大吞吐越高，越占 GPU 显存）
        # 单步「策略可见」观测维度（不含历史堆叠）。
        # 典型组成：ang_vel(3)+gravity(3)+cmd(3)+dof_pos(n)+dof_vel(n)+actions(n)
        num_one_step_observations = 45
        # 策略实际输入 = 单步观测 × 历史帧数（此处堆 6 帧）
        num_observations = num_one_step_observations * 6
        # 单步特权观测（给 Critic）：在策略观测基础上额外加入仿真才有的量
        # +3 base_lin_vel +3 外力扰动 +187 高度扫描点（默认 17×11）
        num_one_step_privileged_obs = 45 + 3 + 3 + 187
        # 特权观测历史长度；1 表示 Critic 只用当前帧特权信息
        # 非 None 时 step() 会返回 privileged_obs_buf（非对称 Actor-Critic）
        num_privileged_obs = num_one_step_privileged_obs * 1
        num_actions = 12  # 动作维数 = 可控关节数（默认 12；轮腿狗常为 16）
        env_spacing = 3.  # 平面网格摆放间距 [m]；heightfield/trimesh 时不用
        send_timeouts = True  # 向算法传递超时标志（超时重置通常不当作失败终止）
        episode_length_s = 20  # 单回合最长时长 [s]

    class terrain:
        """地形网格、摩擦与课程学习参数。"""
        mesh_type = 'trimesh'  # 可选: none / plane / heightfield / trimesh
        horizontal_scale = 0.1  # 水平分辨率 [m/格]
        vertical_scale = 0.005  # 高度分辨率 [m/格]
        border_size = 25  # 地形外围边界宽度 [m]
        curriculum = True  # 是否启用地形难度课程
        static_friction = 1.0  # 静摩擦系数
        dynamic_friction = 1.0  # 动摩擦系数
        restitution = 0.  # 恢复系数（弹性）
        # ---- 粗糙地形高度测量（特权观测 / base_height 奖励用）----
        measure_heights = True  # 是否在机体周围采样地形高度
        # 机体坐标系下采样网格（约 1.6m × 1.0m），点数 17×11=187
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # True 时只生成一种指定地形（配合 terrain_kwargs）
        terrain_kwargs = None  # selected=True 时传入的地形参数字典
        max_init_terrain_level = 5  # 课程开启时，初始可落在的最大难度行
        terrain_length = 8.  # 单个地形块长度 [m]
        terrain_width = 8.  # 单个地形块宽度 [m]
        num_rows = 10  # 地形难度行数（level，由易到难）
        num_cols = 20  # 地形类型列数（type）
        # 各类型占比（旧版随机地形用）：[缓坡, 崎岖坡, 上楼, 下楼, 离散障碍]
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        # ---- 可选：基于 difficulty 的地形混合 ----
        # difficulty ∈ [0,1]：0 全平地，1 按比例混入复杂地形；
        # difficulty < 0 时走旧版/结构化列解锁课程逻辑。
        difficulty = -1.0
        max_rough_terrain_ratio = 0.5  # difficulty=1 时复杂地形占比上限
        # ---- 课程门控：速度跟踪达标才解锁更多地形列 ----
        curriculum_start_type_col = 0  # 起始只开放第 0 列
        curriculum_type_unlock_step = 1  # 每次解锁增加的列数
        command_tracking_pass_threshold = 0.8  # 单环境线速度跟踪达标阈值（相对最大跟踪奖励）
        command_tracking_success_ratio = 0.7  # 达标环境占比超过此值才正常升课
        curriculum_min_timeout_ratio = 0.6  # 正常升课所需存活率（timeout 占比）
        curriculum_unlock_interval_episodes = 3  # 两次解锁之间最小间隔（按 episode 计）
        # 长期卡住时的保底解锁
        curriculum_force_unlock_interval_episodes = 15
        curriculum_force_unlock_min_success_ratio = 0.35
        curriculum_force_unlock_min_timeout_ratio = 0.3
        # trimesh：坡度超过阈值的面会校正为近似竖直面，避免过度倾斜网格
        slope_treshold = 0.75

    class commands:
        """速度/朝向指令采样与指令课程。"""
        curriculum = True  # 是否逐步扩大指令幅值范围
        max_curriculum = 3.0  # 指令课程上限（如 |vx| 最大可扩到该值）
        # 指令维数：vx, vy, yaw_rate, heading（heading 模式下 yaw_rate 由朝向误差算出）
        num_commands = 4
        resampling_time = 10.  # 每隔多少秒重新采样指令 [s]
        heading_command = True  # True：用目标朝向生成偏航角速度；False：直接采样 yaw_rate
        # 指令课程门槛/步长：门槛越高或步长越小，指令变难越慢
        curriculum_pass_threshold = 0.8
        curriculum_step = 0.2

        class ranges:
            """指令采样范围（课程会动态改 lin_vel_x）。"""
            lin_vel_x = [-2.0, 2.0]  # 前向速度 [m/s]
            lin_vel_y = [-1.0, 1.0]  # 侧向速度 [m/s]
            ang_vel_yaw = [-3.14, 3.14]  # 偏航角速度 [rad/s]
            heading = [-3.14, 3.14]  # 目标绝对朝向 [rad]

    class init_state:
        """回合开始时的根状态与默认关节角。"""
        pos = [0.0, 0.0, 1.]  # 初始机身位置 x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # 初始姿态四元数 x,y,z,w
        lin_vel = [0.0, 0.0, 0.0]  # 初始线速度 [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # 初始角速度 [rad/s]
        # action=0 时 PD 跟踪的默认关节目标角；键名需与 URDF 关节名一致
        default_joint_angles = {
            "joint_a": 0.,
            "joint_b": 0.}

    class control:
        """底层控制：动作如何变成关节力矩。"""
        control_type = 'P'  # P=位置PD, V=速度, T=力矩直出
        # PD 增益：按关节名子串匹配（见 LeggedRobot._init_buffers）
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # Kp [N·m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # Kd [N·m·s/rad]
        # 位置控制：目标角 = default + action_scale * action
        action_scale = 0.5
        # 每个策略步对应的仿真子步数；策略 dt = decimation * sim.dt
        decimation = 4

    class asset:
        """机器人资产（URDF）与接触/碰撞相关名称。"""
        file = ""  # URDF 路径，可用 {LEGGED_GYM_ROOT_DIR} 占位
        name = "legged_robot"  # Isaac Gym 中 actor 名称
        foot_name = "None"  # 足端刚体名关键字，用于接触力/足端状态索引
        penalize_contacts_on = []  # 这些连杆触地会进 collision 惩罚
        terminate_after_contacts_on = []  # 这些连杆触地则终止回合（如机身）
        disable_gravity = False
        # 合并由固定关节连接的刚体；个别关节可在 URDF 用 dont_collapse 保留
        collapse_fixed_joints = True
        fix_base_link = False  # True 时固定基座（调试用）
        # GymDofDriveModeFlags: 0 none, 1 pos, 2 vel, 3 effort（力矩）
        default_dof_drive_mode = 3
        self_collisions = 0  # 0 启用自碰撞，1 禁用（按位过滤）
        replace_cylinder_with_capsule = True  # 圆柱碰撞体换成胶囊，更稳更快
        flip_visual_attachments = True  # 部分 mesh 需从 y-up 翻到 z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.  # 关节转子惯量
        thickness = 0.01

    class domain_rand:
        """域随机化：缩小仿真与真机差距。重置或周期性触发。"""
        randomize_payload_mass = True  # 随机负载质量
        payload_mass_range = [-1, 2]  # 相对默认基座质量的增量 [kg]

        randomize_com_displacement = True  # 随机质心偏移
        com_displacement_range = [-0.05, 0.05]  # [m]

        randomize_link_mass = False  # 随机各连杆质量缩放
        link_mass_range = [0.9, 1.1]

        randomize_friction = True  # 随机地面/接触摩擦
        friction_range = [0.25, 1.25]

        randomize_restitution = False  # 随机恢复系数
        restitution_range = [0., 1.0]

        randomize_motor_strength = True  # 随机电机力矩能力缩放
        motor_strength_range = [0.9, 1.1]

        randomize_kp = True  # 随机 PD 的 Kp 缩放因子
        kp_range = [0.9, 1.1]

        randomize_kd = True  # 随机 PD 的 Kd 缩放因子
        kd_range = [0.9, 1.1]

        randomize_initial_joint_pos = True  # 重置时关节角相对默认角随机缩放
        initial_joint_pos_range = [0.5, 1.5]

        disturbance = True  # 周期性对机体施加随机外力
        disturbance_range = [-30.0, 30.0]  # 力分量范围 [N]
        disturbance_interval = 8  # 每隔多少策略步施加一次

        push_robots = True  # 周期性给基座一个随机水平速度脉冲
        push_interval_s = 15  # 推动间隔 [s]
        max_push_vel_xy = 1.  # 最大推动速度 [m/s]

        delay = True  # 模拟执行延迟（动作在 decimation 子步内插值生效）

    class rewards:
        """奖励项权重与相关阈值。

        scales 中非 0 的项会自动绑定 LeggedRobot._reward_<name>()；
        为 0 的项在 _prepare_reward_function 中被移除。
        正数鼓励，负数惩罚（最终还会乘 dt）。
        """

        class scales:
            tracking_lin_vel = 1.5  # 水平速度跟踪（主任务，通常为正）
            tracking_ang_vel = 0.75  # 偏航角速度跟踪
            lin_vel_z = -1.0  # 抑制竖直蹦跳
            ang_vel_xy = -0.05  # 抑制滚转/俯仰角速度
            orientation = -0.5  # 抑制机身倾斜（投影重力 xy 分量）
            base_height = -10.0  # 机身高度贴近目标
            hip_default = -0.5  # 髋关节靠近默认角
            stand_still = -0.5  # 近零指令时少乱动
            collision = -1.0  # 非期望连杆碰撞
            feet_stumble = -0.1  # 足端侧向力过大（绊碰）
            action_rate = -0.01  # 动作变化率（平滑）
            torques = -5.0e-4  # 力矩幅值（近似能耗）
            dof_vel = -1e-7  # 关节速度正则
            dof_acc = -1e-7  # 关节加速度正则

        # True：总奖励截断到 ≥0，减轻早期负奖励导致的学崩
        only_positive_rewards = True
        # 跟踪奖励核：exp(-error^2 / sigma)，sigma 越大对误差越宽容
        tracking_sigma = 0.25
        # 软限位：相对 URDF 限位的可用比例，超出部分进惩罚
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.4  # base_height 奖励的目标高度 [m]
        max_contact_force = 100.  # 足端接触力超过此值开始惩罚 [N]

    class normalization:
        """观测/动作归一化与裁剪。"""
        contact_force_range = [0.0, 50.0]  # 接触力缩放到该区间后再写入观测
        class obs_scales:
            lin_vel = 2.0  # 线速度观测缩放
            ang_vel = 0.25  # 角速度观测缩放
            dof_pos = 1.0  # 关节位置误差缩放
            dof_vel = 0.05  # 关节速度缩放
            height_measurements = 5.0  # 高度扫描缩放
        clip_observations = 100.  # 观测绝对值裁剪上限
        clip_actions = 100.  # 动作绝对值裁剪上限

    class noise:
        """观测噪声（提高真机鲁棒性）。"""
        add_noise = True
        noise_level = 1.0  # 总噪声强度倍率
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        """可视化相机（非 headless 时）。"""
        ref_env = 0  # 参考环境编号
        pos = [10, 0, 6]  # 相机位置 [m]
        lookat = [11., 5, 3.]  # 注视点 [m]

    class sim:
        """Isaac Gym / PhysX 仿真步进参数。"""
        dt = 0.005  # 仿真步长 [s]；策略步长 ≈ dt * control.decimation
        substeps = 1  # 每个 sim step 的物理子步
        gravity = [0., 0., -9.81]  # 重力 [m/s^2]
        up_axis = 1  # 0=Y 向上, 1=Z 向上

        class physx:
            num_threads = 10
            solver_type = 1  # 0: PGS, 1: TGS
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            # GPU 接触对缓冲；环境很多或碰撞复杂时需增大（更占显存）
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            # 0: 不收集接触, 1: 仅最后子步, 2: 所有子步
            contact_collection = 2


class LeggedRobotCfgPPO(BaseConfig):
    """算法侧配置：网络结构、PPO 超参、Runner 日志与断点续训。

    task_registry 会据此创建 HIMOnPolicyRunner + HIMActorCritic + HIMPPO。
    """
    seed = 1  # 随机种子
    runner_class_name = 'HIMOnPolicyRunner'  # 训练循环类名（字符串 eval）

    class policy:
        """Actor / Critic 网络结构。"""
        init_noise_std = 1.0  # 策略高斯噪声初始标准差
        actor_hidden_dims = [512, 256, 128]  # Actor MLP 隐层
        critic_hidden_dims = [512, 256, 128]  # Critic MLP 隐层
        activation = 'elu'  # 激活函数：elu/relu/selu/crelu/lrelu/tanh/sigmoid
        # 仅当使用 ActorCriticRecurrent 时需要：
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        """HIM-PPO / PPO 更新超参。"""
        value_loss_coef = 1.0  # 价值损失权重
        use_clipped_value_loss = True  # 是否对价值函数也做 PPO 式裁剪
        clip_param = 0.2  # 策略比裁剪范围 ε
        entropy_coef = 0.01  # 熵奖励权重（鼓励探索）
        num_learning_epochs = 5  # 每批 rollout 重复更新轮数
        # mini-batch 数；batch_size ≈ num_envs * num_steps_per_env / num_mini_batches
        num_mini_batches = 4
        learning_rate = 1.e-3  # 学习率
        schedule = 'adaptive'  # adaptive：按 KL 调学习率；fixed：固定
        gamma = 0.99  # 折扣因子
        lam = 0.95  # GAE(λ) 系数
        desired_kl = 0.01  # adaptive 模式下的目标 KL
        max_grad_norm = 1.  # 梯度裁剪范数

    class runner:
        """采集长度、迭代次数、保存与恢复。"""
        policy_class_name = 'HIMActorCritic'  # 策略模块类名
        algorithm_class_name = 'HIMPPO'  # 算法类名
        num_steps_per_env = 100  # 每次更新前每个环境采集的步数
        max_iterations = 200000  # 最大策略更新次数

        save_interval = 20  # 每隔多少次迭代保存 checkpoint
        experiment_name = 'test'  # 日志/模型目录名
        run_name = ''  # 可选 run 后缀
        resume = False  # 是否从已有 run 恢复
        load_run = -1  # 加载哪个 run；-1 表示最新
        checkpoint = -1  # 加载哪个 checkpoint；-1 表示最新
        resume_path = None  # 显式恢复路径（优先于 load_run/checkpoint 查找）
