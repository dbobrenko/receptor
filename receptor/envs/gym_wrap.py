from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import gym
import numpy as np
import six
from gym import spaces

import receptor as rf
from receptor.utils import image_preprocess, onehot


class GymWrap(gym.Wrapper):
    """Vectorizes OpenAI Gym spaces."""
    def __init__(self, env):
        # if isinstance(env, six.string_types):
        #     env = gym.make(env)
        super(GymWrap, self).__init__(env)
        if rf.get_random_seed() and hasattr(self.env, 'seed'):
            self.env.seed(rf.get_random_seed())

    def observation(self, obs):
        """Works only with Discrete, MultiBinary and Box."""
        if isinstance(self.observation_space, spaces.Discrete):
            return onehot(obs, self.observation_space.n)
        if isinstance(self.observation_space, spaces.MultiBinary):
            return tuple([onehot(s, 2) for s in obs])
        return obs

    # def action(self, action):
    #     """Works only with Discrete, MultiBinary and Box."""
    #     if isinstance(self.observation_space, spaces.Discrete):
    #         return np.argmax(action)
    #     if isinstance(self.observation_space, spaces.MultiBinary):
    #         return tuple([np.argmax(s) for s in action])
    #     return action

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def _reset(self):
        return self.observation(self.env.reset())


class ImageWrap(gym.Wrapper):
    def __init__(self, env, to_gray=False, new_width=None, new_height=None):
        super(ImageWrap, self).__init__(env=env)
        self._height = new_height
        self._width = new_width
        self._to_gray = to_gray
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        new_shape = list(self.observation_space.shape)
        assert isinstance(self.observation_space, spaces.Box) and 2 <= len(new_shape) <= 3,\
            "Observation space must be 2-D or 3-D tensor."
        new_shape[0] = new_height if new_height else new_shape[0]
        new_shape[1] = new_width if new_width else new_shape[1]
        # Always add channel dimension
        if len(new_shape) == 2:
            new_shape.append(1)

        # Check for grayscale
        if to_gray:
            new_shape[-1] = 1

        self.observation_space = spaces.Box(low=self.observation_space.low.item(0),
                                            high=self.observation_space.high.item(0),
                                            shape=tuple(new_shape),
                                            dtype=self.observation_space.dtype)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def _reset(self):
        return self.observation(self.env.reset())

    def observation(self, obs):
        """Applies such image preprocessing as resizing and converting to grayscale.
        Also, takes maximum value for each pixel value over the current and previous frame.
        Used to get around Atari sprites flickering (see Mnih et al. (2015)).

        Args:
            obs (numpy.ndarray): 2-D or 3-D observation.
        Returns:
            (numpy.ndarray) Preprocessed 3-D observation.
        """
        obs = image_preprocess(obs, resize_height=self._height, resize_width=self._width,
                               to_gray=self._to_gray)
        return obs

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def grayscale(self):
        return self._to_gray


class ActionRepeatWrap(gym.Wrapper):
    """Repeats last action given number of times.
    Args:
        action_repeat (int): The number of step on which the action will be repeated.
    """
    def __init__(self, env, action_repeat):
        super(ActionRepeatWrap, self).__init__(env=env)
        assert action_repeat > 0, "Action repeat number must be higher than 0."
        self._action_repeat = action_repeat

    def _step(self, action):
        obs, reward_total, done, info_all = self.env.step(action)
        for _ in range(self._action_repeat - 1):
            obs, reward, done, info = self.env.step(action)
            reward_total += reward
            info_all.update(info)
            if done:
                break
        return obs, reward_total, done, info_all


class ObservationStackWrap(gym.Wrapper):
    """
    Args:
        obs_stack (int): The length of stacked observations.
            Provided observation_space shape will be automatically modified.
            Doesn't support Tuple spaces.
    """
    def __init__(self, env,  obs_stack):
        super(ObservationStackWrap, self).__init__(env=env)
        assert obs_stack > 0, "Observation stack length must be higher than 0."
        assert not isinstance(self.observation_space, spaces.Tuple),\
            "Observation stack is not compatible with Tuple spaces."
        self.stack_len = obs_stack or 1
        self.observation_space = self.env.observation_space
        new_shape = list(self.observation_space.shape)
        new_shape[-1] = self.observation_space.shape[-1] * obs_stack
        self.observation_space = spaces.Box(low=np.min(self.observation_space.low),
                                            high=np.max(self.observation_space.high),
                                            shape=tuple(new_shape),
                                            dtype=self.observation_space.dtype)
        self._obs_stack = None
        self._last_obs = None

    def _reset(self):
        self._last_obs = self.env.reset()
        return self.reset_stack()

    def _step(self, action):
        self._last_obs, reward, done, info = self.env.step(action)
        self._obs_stack = self.stack_observations(self._last_obs, self.stack_len, self._obs_stack)
        return self._obs_stack, reward, done, info

    def reset_stack(self):
        self._obs_stack = self.stack_observations(self._last_obs, self.stack_len)
        return self._obs_stack

    @staticmethod
    def stack_observations(obs, stack_len, obs_stack=None):
        """Stacks observations along last axis.
           New observations are appended to the existing stack,
           so the chronological order of observations looks like:
           [Obs{N}, Obs{N-1}, ..., Obs{2}, Obs{1: most recent}]

        Args:
            obs (numpy.ndarray): Observation.
            stack_len (int): Stack's total length.
            obs_stack (numpy.ndarray): Current stack of observations.
                If None, passed `obs` will be repeated for `stack_len` times.

        Returns (numpy.ndarray):
            Stacked observations along last axis.
        """
        stack_axis = np.ndim(obs) - 1
        obs_axis_len = np.shape(obs)[stack_axis]
        if obs_stack is None:
            obs_stack = obs
            # np.repeat won't work correctly, since it repeats each element separately,
            # instead of repeating each observation.
            for i in range(stack_len - 1):
                obs_stack = np.append(obs_stack, obs, axis=stack_axis)
        else:
            # Delete the oldest observation.
            # Note, that a single observation may have several depth channels e.g RGB,
            # so that we need to delete each of its channels separately.
            del_indexes = list(range(0, obs_axis_len))
            obs_previous = np.delete(obs_stack, del_indexes, axis=stack_axis)
            obs_stack = np.append(obs_previous, obs, axis=stack_axis)
        assert obs_stack.shape[stack_axis] // obs_axis_len == stack_len
        return obs_stack


class RandomNoOpWrap(gym.Wrapper):
    def __init__(self, env, noop_action, noop_max=30, noop_min=0):
        super(RandomNoOpWrap, self).__init__(env=env)
        assert self.action_space.contains(noop_action),\
            "Invalid action %s for %s environment." % (noop_action, self.env)
        assert noop_max > 0
        assert noop_min >= 0
        self._noop_action = noop_action
        self._noop_max = noop_max
        self._noop_min = noop_min

    def _reset(self):
        self.env.reset()
        skip = np.random.randint(self._noop_min, self._noop_max)
        for _ in range(skip-1):
            # TODO: Skip all wrappers to remove unnecessary preprocessing.
            # obs, _, done, _ = self.unwrapped.step(self._noop_action)
            obs, _, done, _ = self.env.step(self._noop_action)
            if done:
                self.env.reset()
        # Always perform last step with all wrappers applied.
        obs, _, done, _ = self.env.step(self._noop_action)
        return obs


class NormalizeImageWrap(gym.ObservationWrapper):
    def _observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class RewardClipWrap(gym.Wrapper):
    def _step(self, action):
        """Clips reward into {-1, 0, 1} range, as suggested in Mnih et al., 2013."""
        obs, reward, done, info = self.env.step(action)
        info['reward_raw'] = reward
        return obs, np.sign(reward), done, info
