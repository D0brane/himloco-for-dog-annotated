from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class DogRoughCfg(LeggedRobotCfg):

	class env(LeggedRobotCfg.env):
		num_envs = 4096
		num_one_step_observations = 3 + 3 + 3 + 16 + 16 + 16
		num_observations = num_one_step_observations * 6
		num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 11 * 17 + 12
		num_privileged_obs = num_one_step_privileged_obs * 1
		num_actions = 16

	class terrain(LeggedRobotCfg.terrain):
		mesh_type = "trimesh"
		static_friction = 0.8
		dynamic_friction = 0.8
		terrain_proportions = [0.1, 0.1, 0.35, 0.2, 0.25]

	class commands(LeggedRobotCfg.commands):
		curriculum = True
		max_curriculum = 1.5
		num_commands = 4
		resampling_time = 10.0
		heading_command = True

		class ranges:
			lin_vel_x = [-1.0, 1.0]
			lin_vel_y = [-0.6, 0.6]
			ang_vel_yaw = [-1.0, 1.0]
			heading = [-3.14, 3.14]

	class init_state(LeggedRobotCfg.init_state):
		pos = [0.0, 0.0, 0.30]
		default_joint_angles = {
			"rf_leg_joint": 0,
			"rf_bleg_joint": 0.8,
			"rf_sleg_joint": -1.5,
			"rf_foot_joint": 0.0,
			"lf_leg_joint": 0,
			"lf_bleg_joint": 0.8,
			"lf_sleg_joint": -1.5,
			"lf_foot_joint": 0.0,
			"rb_leg_joint": 0,
			"rb_bleg_joint": 0.8,
			"rb_sleg_joint": -1.5,
			"rb_foot_joint": 0.0,
			"lb_leg_joint": 0,
			"lb_bleg_joint": 0.8,
			"lb_sleg_joint": -1.5,
			"lb_foot_joint": 0.0,
		}

	class control(LeggedRobotCfg.control):
		control_type = "P"
		stiffness = {"leg_joint": 50.0, "bleg_joint": 50.0, "sleg_joint": 50.0, "foot_joint": 0.0}
		damping = {"leg_joint": 3.0, "bleg_joint": 3.0, "sleg_joint": 3.0, "foot_joint": 1.0}
		action_scale = 0.0
		decimation = 4
		vel_scale = 12.0
		wheel_direction = -1.0

	class asset(LeggedRobotCfg.asset):
		file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/dog/urdf/dog.urdf"
		name = "dog"
		foot_name = "foot"
		# Keep foot joints out of position tracking terms; useful when foot joints are passive or weakly actuated.
		wheel_name = ["foot_joint"]
		penalize_contacts_on = ["leg_link", "bleg_link", "sleg_link", "base_link"]
		terminate_after_contacts_on = ["base_link"]
		priviledge_contacts_on = ["leg_link", "bleg_link", "sleg_link", "base_link"]
		self_collisions = 1
		replace_cylinder_with_capsule = False
		flip_visual_attachments = False

	class rewards(LeggedRobotCfg.rewards):
		class scales:
			termination = -0.0
			tracking_lin_vel = 1.5
			tracking_ang_vel = 0.75
			wheel_speed = 1.0
			wheel_still = -0.05
			lin_vel_z = -1.0
			ang_vel_xy = -0.05
			orientation = -0.5
			base_height = -10.0
			hip_default = -0.5
			stand_still = -1.0
			collision = -1.5
			feet_stumble = -0.2
			action_rate = -0.01
			torques = -2.0e-4
			dof_vel = -1e-7
			dof_acc = -1e-7
			run_still = 0.0
			foot_clearance = 0.0
			joint_power = -1e-5
			smoothness = -0.01
			feet_air_time = 0.0
			hip_pos = -0.0
			stumble = -0.0
			diagonal_phase = 0.0

		only_positive_rewards = True
		tracking_sigma = 0.25
		soft_dof_pos_limit = 1.0
		soft_dof_vel_limit = 1.0
		soft_torque_limit = 1.0
		base_height_target = 0.3
		clearance_height_target = 0.08
		max_contact_force = 100.0
		base_height_curriculum = True  # 是否启用高度课程学习。
		base_height_curriculum_steps = 8000000  # 高度课程学习总步数。
		base_height_target_min = 0.26  # 高度课程下限。
		base_height_target_max = 0.38  # 高度课程上限。
		base_height_start_span = 0.01  # 高度课程初始跨度。

class DogRoughCfgPPO(LeggedRobotCfgPPO):
	class algorithm(LeggedRobotCfgPPO.algorithm):
		entropy_coef = 0.005

	class runner(LeggedRobotCfgPPO.runner):
		save_interval = 10
		num_steps_per_env = 48
		max_iterations = 20000
		experiment_name = "dog"
		run_name = ""
		resume = None
		load_run = -1
		checkpoint = -1
		resume_path = None
