from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import time

import gym
import six
import torch
from six.moves import range  # pylint: disable=redefined-builtin
from tensorboardX import SummaryWriter
from torch import multiprocessing as mp

from receptor import logger
from receptor.core.rollout import Rollout
from receptor.core.stats import Stats, flush_stats


@six.add_metaclass(abc.ABCMeta)
class BaseAgent(object):
    @abc.abstractmethod
    def __init__(self, net, device='cuda:0', logdir='', name='', *args, **kwargs):
        """Abstract base class for Deep Network-based agents.

        Args:
            env (gym.Env): Environment instance.
            net (models.Model): Model builder.
            device (str): TensorFlow device.
            name (str): Agent's name prefix.
        """
        super(BaseAgent, self).__init__()
        self.net = net
        self.name = name
        self.logdir = logdir
        self.device = device if torch.cuda.is_available() else "cpu"
        self.obs_step = mp.Value('l', 0)
        self.train_step = mp.Value('l', 0)
        self.episode_step = mp.Value('l', 0)
        self.writer = None if logdir is None else SummaryWriter(logdir)

    def train_on_batch(self, rollout, lr=None, aux_losses=(), summarize=False, importance=None):
        """Performs optimization with given rollout batch.

        Args:
            rollout (core.Rollout): Rollout from environment.
            lr (float): Learning rate. Pass None to leave constant lr.
            aux_losses (tuple, list): Auxiliary Losses.
            summarize (bool): Whether to write logs.
            importance (list): Importance weight sampling.

        Returns: Loss value.
        """
        raise NotImplementedError

    def predict_on_batch(self, obs_batch):
        """Computes network output for given batch of observations."""
        raise NotImplementedError

    def act(self, obs):
        """Computes greedy action for given observation.
        Args:
            obs (numpy.ndarray): Observation.
        Returns:
            Action.
        """
        raise NotImplementedError

    def explore(self, obs):
        """Computes action in exploration mode for given observation.
        Args:
            obs (numpy.ndarray): Observation.
        Returns:
            Action.
        """
        raise NotImplementedError

    def get_state(self):
        return {'net': self.net,
                'obs_step': self.obs_step.value,
                'train_step': self.train_step.value,
                'episode_step': self.episode_step.value,
                }

    def set_state(self, state):
        self.net = state['net']
        self.obs_step.value = state['obs_step']
        self.train_step.value = state['train_step']
        self.episode_step.value = state['episode_step']

    def save(self, path, model_name='agent.pt'):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.get_state(), os.path.join(path, model_name))
        logger.info('Agent has been saved to: %s' % os.path.join(path, model_name))

    def load(self, checkpoint):
        if not os.path.exists(checkpoint):
            raise ValueError('Checkpoint path does not exists: %s' % checkpoint)
        state = torch.load(checkpoint)
        self.set_state(state)
        logger.info('Agent has been restored from: %s' % checkpoint)

    def test(self, env, episodes, maxsteps=1e5, render=False, fpslimit=None, writer=None,
             name=None):
        """Tests agent's performance on a given number of episodes.

        Args:
            env (gym.Env): Test environment.
            episodes (int): Number of episodes.
            maxsteps (int): Maximum allowed step per episode.
            render (bool): Enables game screen rendering.
            fpslimit (int): Maximum allowed fps. To disable fps limitation, pass None.
            writer (FileWriter): TensorBoard summary writer.
            name (str): Log prefix, used in console and TensorBoard logging.
                If None, 'self.name/Test' is used.
        """
        stats = Stats(agent=self)
        delta_frame = 1. / fpslimit if fpslimit else 0
        step_counter = 0
        episode_counter = 0
        maxsteps = int(maxsteps)
        for _ in range(episodes):
            obs = env.reset()
            for i in range(maxsteps):
                start_time = time.time()
                action = self.act(obs)
                obs, r, terminal, info = env.step(action)
                step_limit = i >= maxsteps - 1
                terminal = terminal or step_limit
                if step_limit:
                    logger.info("Interrupting test episode due to the "
                                "maximum allowed number of steps (%d)" % i)
                step_counter += 1
                episode_counter += terminal
                stats.add_rewards(r, terminal, info)
                if self.writer is not None:
                    self.writer.add_scalar('test/R', r)
                if render:
                    env.render()
                    if delta_frame > 0:
                        delay = max(0, delta_frame - (time.time() - start_time))
                        time.sleep(delay)
                if terminal:
                    break
        name = '%s Test' % self.name if None else name
        flush_stats(stats, log_progress=False, log_performance=False, log_hyperparams=False,
                    name=name, writer=writer)
