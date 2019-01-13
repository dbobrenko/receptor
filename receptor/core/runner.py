from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from six.moves import range  # pylint: disable=redefined-builtin

from receptor.core.rollout import Rollout, RolloutParallel


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
        self.cache_output = cache_output

    def sample(self, gamma=0.99):
        rollouts = Rollout()
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
        with torch.no_grad():
            actions, output = self.agent.explore_on_batch(self.current_obs)
            expected_values = output['value'].cpu().numpy().squeeze(-1)
        rollouts.discount_rewards(expected_values, gamma=gamma)
        return rollouts


class ParallelEnvRunner(EnvRunner):
    def sample(self, gamma=0.99):
        rollouts = RolloutParallel()
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
            rollouts.add(self.current_obs, act, reward, term, output, info)
            self.current_obs = obs_next

        with torch.no_grad():
            actions, output = self.agent.explore_on_batch(self.current_obs)
            expected_values = output['value'].cpu().numpy().squeeze(-1)
        rollouts.discount_rewards(expected_values, gamma=gamma)
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
        with torch.no_grad():
            actions, output = self.agent.explore_on_batch(rollouts.obs_next)
            expected_values = output['value'].cpu().numpy().squeeze(-1)
        rollouts.discount_rewards(expected_values, gamma=gamma)
        return rollouts
