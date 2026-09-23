from legged_gym.envs import LeggedRobot

from .dog_config_change import DogRoughCfg


class Dog(LeggedRobot):
	cfg: DogRoughCfg
