from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
from termcolor import colored
from torch import multiprocessing as mp

from receptor import logger


def _make_row(*column_messages, **kwargs):
    """Makes a formatted string.

    Args:
        *column_messages (str): Messages.
        **kwargs:

    Returns:

    """
    color = kwargs.get("color", None)
    col_size = kwargs.get("column_size", 40)
    message = ""
    for m in column_messages:
        if m is not None:
            message += str("\t\t%-" + str(col_size) + "s ") % str(m)
    return colored(message, color=color)


def _log_rows(*rows):
    table = ""
    for r in rows:
        if r is not None:
            table += "%s\n" % r
    logger.info(table)


def flush_stats(stats, name, log_progress=True, log_rewards=True, log_performance=True,
                log_hyperparams=True, maxsteps=None, writer=None):
    """Prints agent's logs into the console and TensorBoard. Made as a separate function,
    in order to make possible logging from list of Stats (e.g. in A3C agent).

    Args:
        stats (Stats or list of Stats):
        name (str): Prefix name, will be used in TensorBoard and terminal logs.
        log_progress (bool): Enables progress (steps/episodes passed) logging.
        log_rewards (bool): Enables reward statistics logging.
        log_performance (bool): Enables performance (fps) logging.
        log_hyperparams (bool): TODO Enables hyperparameter logging.
        maxsteps (int): Total training steps. If None, disables progress percentage.
        writer (tensorboardx.SummaryWriter): Summary writer.

    Returns:

    """
    name = stats.agent.name if name is None else name
    if isinstance(stats, Stats):
        stats = [stats]
    stat = stats[0]
    delta_time = time.time() - stat.last_time.value
    optim_per_sec = (stat.agent.train_step.value - stat.last_optimize.value) / delta_time
    steps = stat.agent.obs_step.value
    episodes = stat.agent.episode_step.value
    obs_per_sec = (stat.agent.obs_step.value - stat.last_step.value) / delta_time
    reward_step = 0
    lr = 0
    episode_rewards = []
    exploration = 0
    for stat in stats:
        reward_step += stat.reward_stats.reset_step_rewards()
        if stat.reward_stats.episode.value > 0:
            episode_rewards.append(stat.reward_stats.reset_episode_rewards())
        # exploration += stat.agent.gamma
        # lr += stat.lr
        stat.last_time.value = int(time.time())
        stat.last_step.value = stat.agent.obs_step.value
        stat.last_optimize.value = stat.agent.train_step.value

    reward_step /= len(stats)
    reward_ep = float(np.mean(episode_rewards or 0))
    exploration /= len(stats)
    lr /= len(stats)

    percent = "(%.2f%%)" % (100 * (steps / maxsteps)) if maxsteps is not None else ""
    _log_rows(colored(name, color='green', attrs=['bold']),
              _make_row('%-20s %d %s' % ('Steps', steps, percent),
                        '%-20s %d' % ('Episodes', episodes),
                        color='blue') if log_progress else None,

              _make_row('%-20s %.4f' % ('Reward/Step', reward_step),
                        '%-20s %.2f' % ('Reward/Episode', reward_ep) if reward_ep else None,
                        color='blue') if log_rewards else None,

              _make_row('%-20s %.2f' % ('Observation/Sec', obs_per_sec),
                        '%-20s %.2f' % ('Optimization/Sec', optim_per_sec),
                        color='cyan') if log_performance else None,

              # _make_row('%-20s %.2f' % ('Exploration Rate', exploration),
              #           '%-20s %.2e' % ('Learning Rate', lr),
              #           ) if log_hyperparams else None
              )

    if writer is not None:
        writer.add_scalar(name+'/TotalEpisodes', episodes, steps)
        writer.add_scalar(name+'/ObsPerSec', obs_per_sec, steps)
        writer.add_scalar(name+'/OptimizePerSec', optim_per_sec, steps)
        writer.add_scalar(name+'/RewardPerStep', reward_step, steps)
        if reward_ep > 0:
            writer.add_scalar(name+'/RewardPerEpisode', reward_ep, steps)


class Stats(object):
    def __init__(self, agent, logdir=''):
        """Stats logger.

        Args:
            agent (Agent): Agent instance.
        """
        self.agent = agent
        self.reward_stats = RewardStats()
        self.last_time = mp.Value('l', int(time.time()))
        self.last_step = mp.Value('l', self.agent.obs_step.value)
        self.last_optimize = mp.Value('l', self.agent.train_step.value)

    def add_rewards(self, rewards, terms, info=None):
        self.reward_stats.add(rewards, terms)

    def add(self, rollout):
        """Adds statistics. Expected to be called after each `gym.Env.step`.

        Args:
            rollout (Rollout): Rollout.
        """
        # rewards = [info.get('reward_raw', reward) for reward, info in zip(rewards, infos)]
        self.reward_stats.add(rollout.rewards, rollout.terms)

    def flush(self, name=None):
        flush_stats(self, name)


class RewardStats(object):
    """Keeps agent's step and episode reward statistics."""
    def __init__(self):
        self.episode_sum = mp.Value('f', 0.0)
        self.step_sum = mp.Value('f', 0.0)
        self._running_ep_r = mp.Value('f', 0.0)
        self.step = mp.Value('l', 0)
        self.episode = mp.Value('l', 0)
        self.episode_min = mp.Value('f', float('+inf'))
        self.episode_max = mp.Value('f', float('-inf'))

    def add(self, reward, terminal):
        """Adds reward and terminal state (end of episode).
        Args:
            reward (float, np.ndarray or list): Reward.
            terminal (bool, np.ndarray or list): Whether the episode was ended.
        """
        self.step.value += np.size(reward)
        self.step_sum.value += np.sum(reward)
        self._running_ep_r.value += np.sum(reward)
        # Episode rewards book keeping
        if np.any(terminal):
            self.episode_sum.value += self._running_ep_r.value
            if self._running_ep_r.value < self.episode_min.value:
                self.episode_min.value = self._running_ep_r.value
            if self._running_ep_r.value > self.episode_max.value:
                self.episode_max.value = self._running_ep_r.value
            self._running_ep_r.value = 0
            self.episode.value += np.sum(terminal)

    def add_batch(self, reward_batch, terminal_batch):
        """Adds batch with rewards and terminal states (end of episode).
        Args:
            reward_batch: List with rewards after each action.
            terminal_batch: List with booleans indicating the end of the episode after each action.
        """
        assert len(reward_batch) == len(terminal_batch)
        if not np.any(terminal_batch):
            sum_batch = np.sum(reward_batch)
            self.step.value += len(reward_batch)
            self.step_sum.value += sum_batch
            self._running_ep_r.value += sum_batch
            return
        # If batch contains terminal state, add by element
        for reward, term in zip(reward_batch, terminal_batch):
            self.add(reward, term)

    def step_average(self):
        """Computes average reward per step."""
        return self.step_sum.value / (self.step.value or 1)

    def episode_average(self):
        """Computes average reward per episode."""
        return (self.episode_sum.value / self.episode.value) if self.episode.value else 0

    def reset_step_rewards(self):
        """Resets step counters.
        Returns: Average reward per step.
        """
        step = self.step_average()
        self.step_sum.value = 0.0
        self.step.value = 0
        return step

    def reset_episode_rewards(self):
        """Resets episode counters.
        Returns: Average reward per episode.
        """
        ep = self.episode_average()
        self.episode_sum.value = 0.0
        self.episode.value = 0
        self.episode_min.value = float('+inf')
        self.episode_max.value = float('-inf')
        return ep

    def reset(self):
        """Resets all counters.
        Returns: Average reward per step, Average reward per episode.
        """
        step = self.reset_step_rewards()
        ep = self.reset_episode_rewards()
        return step, ep
