import gym
from gym import spaces


class GymToTorch(gym.ObservationWrapper):
    """Converts HWC observation space into CHW"""
    def __init__(self, env=None):
        super(GymToTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low.item(0),
            self.observation_space.high.item(0),
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class TorchToGym(gym.ObservationWrapper):
    """Converts CHW observation space into HWC"""
    def __init__(self, env=None):
        super(TorchToGym, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low.item(0),
            self.observation_space.high.item(0),
            [obs_shape[1], obs_shape[2], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(1, 2, 0)

