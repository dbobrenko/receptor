from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from receptor import logger
from receptor.core import Stats
from receptor.core.runner import ParallelEnvRunner
from receptor.core.schedule import Schedule
from receptor.core.stats import flush_stats
from receptor.trainers.trainer import BaseTrainer
from receptor.utils import torch_utils
from tensorboardX import SummaryWriter


class SyncTrainer(BaseTrainer):
    def __init__(self, agent, thread_envs, maxsteps, batch_size,
                 logdir, logfreq, log_on_term=True, lr_schedule=None,
                 render=False, test_env=None, test_render=False, gamma=0.99,
                 test_episodes=1, test_maxsteps=5e4):
        """Creates trainer based on Experience Replay buffer.

        Args:
            agent:
            maxsteps (int): Total amount of seen observations.
            logdir (str): Path used for summary and checkpoints.
            logfreq (int): Checkpoint and summary saving frequency (in seconds).
            log_on_term (bool): Whether to log only after episode ends.
            lr_schedule (core.Schedule): Learning rate scheduler.
            render (bool): Enables game screen rendering.
            test_env (gym.Env): Environment instance, used for testing.
            test_render (bool): Enables rendering for test evaluations.
            test_episodes (int): Number of test episodes. To disable test eval  uation, pass 0.
            test_maxsteps (int): Maximum step allowed during test per episode.
        """
        self.name = "Sync" + agent.name
        self.agent = agent
        self.maxsteps = maxsteps
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.logdir = logdir
        self.logfreq = logfreq
        self.log_on_term = log_on_term
        self.render = render
        self.test_env = test_env
        self.test_render = test_render
        self.test_episodes = test_episodes
        self.test_maxsteps = test_maxsteps
        self._last_log_time = time.time()
        self.envs = thread_envs
        self.gamma = gamma
        self.stats = Stats(self.agent)

    def train(self):
        """Starts training."""
        writer = SummaryWriter(self.logdir)
        last_log_time = time.time()
        lr_schedule = Schedule.create(self.lr_schedule,
                                      torch_utils.get_lr(self.agent.opt),
                                      self.maxsteps)
        runner = ParallelEnvRunner(agent=self.agent, env=self.envs,
                                   batch_size=self.batch_size)

        stats = Stats(self.agent)
        try:
            while self.agent.obs_step.value < self.maxsteps:
                rollout = runner.sample(self.gamma)
                lr = lr_schedule.value(self.agent.obs_step.value)
                self.agent.train_on_batch(rollout, lr=lr, summarize=False)
                stats.add(rollout)

                if self.render:
                    self.envs.render()
                if time.time() - last_log_time >= self.logfreq:
                    last_log_time = time.time()
                    flush_stats(stats, name="%s/Train" % self.name,
                                maxsteps=self.maxsteps, writer=writer)
                    self.agent.save(self.logdir)

                    if self.test_env is not None:
                        self.agent.test(self.test_env,
                                        self.test_episodes,
                                        maxsteps=self.test_maxsteps,
                                        render=self.test_render,
                                        name="%s/Test" % self.name,
                                        writer=writer)
        except KeyboardInterrupt:
            logger.info('Caught Ctrl+C! Stopping training process.')
        writer.close()
        logger.info('Saving progress & performing evaluation.')
        self.agent.save(self.logdir)
        if self.test_env is not None:
            self.agent.test(self.test_env, self.test_episodes, render=self.test_render)
        logger.info('Training finished!')

    def save(self):
        pass

    def load(self):
        pass
