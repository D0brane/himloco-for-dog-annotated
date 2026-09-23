from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class DogRoughCfg(LeggedRobotCfg):

	class env(LeggedRobotCfg.env):
		num_envs = 2048  # 并行仿真环境数量（降低以减小GPU碰撞对内存压力）。
		num_one_step_observations = 3 + 3 + 3 + 16 + 16 + 16  # 单步观测维度：角速度+重力投影+指令+关节位置/速度+上一时刻动作。
		num_observations = num_one_step_observations * 6  # 策略输入使用6步历史堆叠。
		num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 11 * 17 + 12  # 给critic的额外特权状态信息维度。
		num_privileged_obs = num_one_step_privileged_obs * 1  # 特权观测历史长度（当前为1帧）。
		num_actions = 16  # 动作维度，等于可控关节总数。

	class terrain(LeggedRobotCfg.terrain):
		mesh_type = "trimesh"  # 使用三角网格地形。
		static_friction = 0.8  # 接触模型静摩擦系数。
		dynamic_friction = 0.8  # 接触模型动摩擦系数。
		num_cols = 9  # 课程列数，与下方9个课程阶段一一对应。
		terrain_proportions = [1.0]  # 结构化课程下该项不再参与课程列类型分配。
		curriculum_terrain_types = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 0平地,1平地+小起伏,2平地+大起伏,3斜坡,4斜坡+小起伏,5斜坡+大起伏,6下楼,7上楼,8离散障碍。
		max_init_terrain_level = 0  # 课程开始时仅在最简单行初始化。
		curriculum_start_type_col = 0  # 从最容易地形列开始。
		curriculum_type_unlock_step = 1  # 每次课程晋级解锁1列新地形。
		command_tracking_pass_threshold = 0.82  # 单环境速度跟踪达标阈值（适中，避免卡住）。
		command_tracking_success_ratio = 0.6  # 达标环境占比超过该值才进入下一课程。
		curriculum_min_timeout_ratio = 0.7  # 常规升课所需存活率（timeout占比）。
		curriculum_unlock_interval_episodes = 3  # 地形解锁最小间隔（episode）。
		curriculum_force_unlock_interval_episodes = 12  # 长时间卡住时允许保底解锁。
		curriculum_force_unlock_min_success_ratio = 0.35  # 触发保底解锁的最低达标占比。
		curriculum_force_unlock_min_timeout_ratio = 0.5  # 触发保底解锁的最低存活率。

	class commands(LeggedRobotCfg.commands):
		curriculum = True  # 训练中逐步提升指令难度。
		max_curriculum = 2.5  # 课程系数上限（增大后可采样更高前向速度）。
		curriculum_pass_threshold = 0.82  # 指令课程通过阈值（适中）。
		curriculum_step = 0.15  # 每次指令范围扩展步长（避免过慢）。
		num_commands = 4  # 指令维度：[vx, vy, yaw_rate, heading]。
		resampling_time = 10.0  # 每隔多少秒重采样一次目标指令。
		heading_command = True  # 根据目标朝向生成偏航控制指令。

		class ranges:
			lin_vel_x = [-2, 2]  # 前向速度指令范围（m/s），增大命令速度输入。
			lin_vel_y = [-1.0, 1.0]  # 侧向速度指令范围（m/s）。
			ang_vel_yaw = [-1.0, 1.0]  # 偏航角速度指令范围（rad/s）。
			heading = [-3.14, 3.14]  # 绝对朝向目标范围（rad）。

	class init_state(LeggedRobotCfg.init_state):
		pos = [0.0, 0.0, 0.45]  # 初始机身位置 [x, y, z]（世界坐标系）。
		default_joint_angles = {
			"rf_leg_joint": -0.05,
			"rf_bleg_joint": 0.8,
			"rf_sleg_joint": 1.5,
			"rf_foot_joint": 0.0,
			"lf_leg_joint": 0.05,
			"lf_bleg_joint": -0.8,
			"lf_sleg_joint": -1.5,
			"lf_foot_joint": 0.0,
			"rb_leg_joint": -0.05,
			"rb_bleg_joint": 0.8,
			"rb_sleg_joint": 1.5,
			"rb_foot_joint": 0.0,
			"lb_leg_joint": 0.05,
			"lb_bleg_joint": -0.8,
			"lb_sleg_joint": -1.5,
			"lb_foot_joint": 0.0,
		}  # PD控制下 action=0 时对应的默认目标关节姿态。

	class control(LeggedRobotCfg.control):
		control_type = "P"  # 位置型PD控制。
		stiffness = {"leg_joint": 40.0, "bleg_joint": 40.0, "sleg_joint": 40.0, "foot_joint": 0.0}  # 关节空间Kp增益。
		damping = {"leg_joint": 1.0, "bleg_joint": 1.0, "sleg_joint": 1.0, "foot_joint": 0.5}  # 关节空间Kd增益。
		action_scale = 0.25  # 将归一化动作映射为关节目标偏移量。
		decimation = 4  # 每个策略动作对应的仿真步数。
		vel_scale = 10.0  # 与轮/足端速度相关项使用的缩放系数。
		wheel_direction = -1.0  # 轮式关节方向符号约定。
		# Runtime scale applied to torque limits read from URDF. Change to <1.0 to
		# reduce allowed torques without editing the URDF. E.g. 0.5 halves the
		# torque limits.
		torque_limit_scale = 1.0

	class asset(LeggedRobotCfg.asset):
		file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/dog/urdf/dog.urdf"  # 机器人URDF路径。
		name = "dog"  # 仿真器中的机器人名称。
		foot_name = "foot"  # 足端/末端执行器对应的刚体名称关键字。
		wheel_name = ["foot_joint"]  # 被当作轮式关节处理的关节名。
		penalize_contacts_on = ["leg_link", "bleg_link", "sleg_link", "base_link"]  # 发生接触会被惩罚的连杆。
		terminate_after_contacts_on = ["base_link"]  # 这些连杆触地时终止回合。
		priviledge_contacts_on = ["leg_link", "bleg_link", "sleg_link", "base_link"]  # 暴露到特权观测中的接触连杆。
		self_collisions = 1  # 1表示按URDF过滤配置关闭自碰撞对。
		replace_cylinder_with_capsule = False  # 保持原始碰撞体形状，不替换为胶囊体。
		flip_visual_attachments = False  # 不翻转导入网格的可视化附件。

	class rewards(LeggedRobotCfg.rewards):
		class scales:
			tracking_lin_vel = 3.0  # 线速度跟踪奖励。
			tracking_ang_vel = 0.75  # 偏航角速度跟踪奖励。
			wheel_speed = 0.5  # 在有平移指令时鼓励轮子转动。
			wheel_still = -0.05  # 在接近平移零指令时惩罚轮子空转。
			leg_motion = -0.1  # 在有平移指令时惩罚腿部关节运动，鼓励以轮式推进为主。
			lin_vel_z = -1.0  # 惩罚机身竖直方向速度。
			ang_vel_xy = -0.05  # 惩罚滚转/俯仰角速度。
			orientation = -0.8  # 惩罚机身偏离直立姿态。
			base_height = -10.0  # 惩罚机身高度偏离目标值。
			hip_default = -0.8  # 惩罚髋关节偏离默认姿态。
			stand_still = -0.5  # 在近零指令下抑制多余运动。
			collision = -1.2  # 惩罚非期望碰撞。
			feet_stumble = -0.1  # 惩罚足端绊碰/拖拽事件。
			action_rate = -0.03  # 动作变化率平滑惩罚。
			torques = -5.0e-4  # 力矩幅值惩罚（近似能耗约束）。
			dof_vel = -1e-7  # 关节速度正则项。
			dof_acc = -1e-7  # 关节加速度正则项。
			run_still = -0.05  # 在运动指令下保持静止的惩罚。

		only_positive_rewards = True  # 将总奖励截断为非负，降低早期训练崩溃风险。
		tracking_sigma = 0.25  # 速度跟踪指数奖励核宽度。
		soft_dof_pos_limit = 1.0  # 软关节位置限位系数。
		soft_dof_vel_limit = 1.0  # 软关节速度限位系数。
		soft_torque_limit = 1.0  # 软力矩限位系数。
		base_height_target = 0.4  # base_height项对应的目标机身高度。
		max_contact_force = 100.0  # 超过该阈值的接触力会被惩罚。

	class sim(LeggedRobotCfg.sim):
		class physx(LeggedRobotCfg.sim.physx):
			# NOTE: lowered these values from the original defaults to avoid
			# excessive GPU PhysX memory allocations on machines with limited
			# GPU memory. If you have a workstation with ample GPU RAM you can
			# increase these again.
			# 原始值会在低显存卡上触发大块GPU分配，导致分配失败/段错误。
			max_gpu_contact_pairs = 2**22  # was 2**25
			default_buffer_size_multiplier = 4  # was 16


class DogRoughCfgPPO(LeggedRobotCfgPPO):
	class algorithm(LeggedRobotCfgPPO.algorithm):
		entropy_coef = 0.005  # 熵正则权重，用于鼓励探索。

	class runner(LeggedRobotCfgPPO.runner):
		save_interval = 50  # 每多少次迭代保存一次检查点。
		num_steps_per_env = 48  # 每次更新前每个环境采样的时间步长度。
		max_iterations = 20000  # PPO最大训练迭代次数。
		experiment_name = "dog"  # 日志目录中的实验名。
		run_name = ""  # 可选的运行名后缀。
		resume = None  # 是否恢复训练（具体行为由框架实现决定）。
		load_run = -1  # 加载的run编号，-1表示最新。
		checkpoint = -1  # 加载的checkpoint编号，-1表示最新。
		resume_path = None  # 显式恢复路径，优先于run/checkpoint查找。
