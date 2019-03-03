from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import six

from receptor.envs import ObservationStackWrap, ActionRepeatWrap, \
    GymWrap, RandomNoOpWrap, RewardClipWrap, ImageWrap, NormalizeImageWrap


class FireResetWrap(gym.Wrapper):
    def __init__(self, env):
        super(FireResetWrap, self).__init__(env=env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


class AtariLifeReset(FireResetWrap):
    """Starts new episode after Atari live lost.
    Starts each episode from fire_action."""
    def __init__(self, env):
        super(AtariLifeReset, self).__init__(env=env)
        self.env = FireResetWrap(self.env)
        assert hasattr(self.env.unwrapped, 'ale')
        self._lives = 0
        self.needs_reset = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.needs_reset = done
        lives = self.env.unwrapped.ale.lives()
        # If live is lost reset episode
        if self._lives > lives > 0:
            if not done:
                info['live_lost'] = True
            done = True
        self._lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.needs_reset:
            self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        self._lives = self.env.unwrapped.ale.lives()
        return obs


class AtariObsStackLifeReset(gym.Wrapper):
    """Resets environment and observation stack after Atari live lost.
    Starts each episode from fire_action.
    """
    def __init__(self, env, obs_stack, life_reset=True):
        super(AtariObsStackLifeReset, self).__init__(env=env)
        # self.life_env = AtariLifeReset(self.env)
        self.env = ObservationStackWrap(self.env, obs_stack)
        self.needs_reset = True
        self._lives = 0
        self._life_reset = life_reset
        assert hasattr(self.env.unwrapped, 'ale')
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        # Stack observations
        obs, reward, done, info = self.env.step(action)
        self.needs_reset = done
        lives = self.env.unwrapped.ale.lives()
        # If live is lost reset episode
        if self._lives > lives > 0:
            info['live_lost'] = True
            if not done:
                obs = self.env.reset_stack()
            if self._life_reset:
                done = True
            else:
                obs = self._start_button()
        self._lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.needs_reset:
            self.env.reset()
        obs = self._start_button(**kwargs)
        self._lives = self.env.unwrapped.ale.lives()
        return obs

    def _start_button(self, **kwargs):
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


class ActionRepeatMaxWrap(ActionRepeatWrap):
    """Repeats last action given number of times and applies max pooling to the last two frames.
    Args:
        action_repeat (int): The number of step on which the action will be repeated.
    """
    def __init__(self, env, action_repeat):
        super(ActionRepeatMaxWrap, self).__init__(env=env, action_repeat=action_repeat)

    def _step(self, action):
        obs, reward_total, done, info_all = self.env.step(action)
        last_obses = [obs, obs]
        for _ in range(self._action_repeat - 1):
            obs, reward, done, info = self.env.step(action)
            reward_total += reward
            info_all.update(info)
            last_obses[0] = last_obses[1]
            last_obses[1] = obs
            if done:
                break
        # Takes maximum value for each pixel value over the current and previous frame.
        # Used to get around Atari sprites flickering (Mnih et al. (2015))
        if self._action_repeat > 1:
            obs = np.maximum.reduce(last_obses)
        return obs, reward_total, done, info_all


class AtariWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 life_reset=True,
                 noop_action=None,
                 action_repeat=1,
                 obs_stack=4,
                 to_gray=True,
                 new_width=84,
                 new_height=84,
                 clip_rewards=True):
        if isinstance(env, six.string_types):
            env = gym.make(env)
        super(AtariWrapper, self).__init__(env=env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.env = GymWrap(self.env)
        if noop_action:
            self.env = RandomNoOpWrap(self.env, noop_action=noop_action)
        self.env = ActionRepeatMaxWrap(self.env, action_repeat=action_repeat)
        if clip_rewards:
            self.env = RewardClipWrap(self.env)
        self.env = ImageWrap(self.env, to_gray=to_gray, new_width=new_width,
                             new_height=new_height)
        # self.env = NormalizeImageWrap(self.env)
        self.env = AtariObsStackLifeReset(self.env, obs_stack=obs_stack, life_reset=life_reset)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range

    def reset(self):
        return self.env.reset()
