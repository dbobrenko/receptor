from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from six.moves import range  # pylint: disable=redefined-builtin

from receptor.core.rollouts import Rollouts, ParallelRollouts


class EnvRunner(object):
    def __init__(self, agent, env, batch_size, cache_output=False):
        """Wrapper for environment batch sampling.

        Args:
            agent (Agent): Learning agent.
            env (gym.Env): Environment with gym-like interface.
            batch_size (int): Batch size.
        """
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self.current_obs = None
        self.obs_next = None
        self.cache_output = cache_output

    def sample(self, gamma=0.99):
        rollouts = Rollouts()
        if self.current_obs is None:
            self.current_obs = self.env.reset()
        for i in range(self.batch_size):
            if self.cache_output:
                act, output = self.agent.explore_on_batch([self.current_obs])
            else:
                with torch.no_grad():
                    act, output = self.agent.explore_on_batch([self.current_obs])
                    output = None
            obs_next, reward, term, info = self.env.step(act)
            rollouts.add(self.current_obs, act, reward, term, output, info)
            self.current_obs = obs_next
            if term:
                self.current_obs = self.env.reset()
                break
        rollouts.obs_next = self.current_obs
        return rollouts


class ParallelEnvRunner(EnvRunner):
    def sample(self, gamma=0.99):
        rollouts = ParallelRollouts()
        if self.current_obs is None:
            self.current_obs = self.env.reset()
        for i in range(self.batch_size):
            if self.cache_output:
                act, output = self.agent.explore_on_batch(self.current_obs)
            else:
                with torch.no_grad():
                    act, output = self.agent.explore_on_batch(self.current_obs)
                    output = None
            obs_next, reward, term, info = self.env.step(act)
            rollouts.add(self.current_obs, act, reward, term, info)
            self.current_obs = obs_next
        rollouts.obs_next = self.current_obs
        return rollouts


class ReplayRunner(object):
    def __init__(self, agent, env, replay, shuffle=False):
        """Adapter for batch sampling from environments and replays.

        Args:
            agent (Agent): Learning agent.
            env (gym.Env): Environment with gym-like interface.
            replay (Replay): Replay for training. To disable training from replay, pass None.
        """
        self.agent = agent
        self.env = env
        self.replay = replay
        self.replay.shuffle = shuffle
        self._obs = None

    def sample(self, gamma=0.99):
        """Samples batch from given data provider.
        Increments agent's step and episode counters.

        Returns:
            Batch with replay: (obs, action, reward, terminal, next-obs).
        """
        rollouts = self.replay.sample()
        return rollouts
