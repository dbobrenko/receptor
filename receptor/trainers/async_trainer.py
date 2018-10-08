from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from threading import Lock
from threading import Thread

from torch import multiprocessing as mp

from receptor import logger
from receptor.core import Stats
from receptor.core.runner import EnvRunner, ParallelEnvRunner
from receptor.core.schedule import Schedule
from receptor.core.stats import flush_stats
from receptor.envs.parallel_envs import CloudpickleWrapper
from receptor.trainers.trainer import BaseTrainer
from receptor.utils import torch_utils


class AsyncTrainer(BaseTrainer):
    def __init__(self, agent, thread_agents, thread_env, maxsteps, batch_size,
                 logdir, logfreq, log_on_term=True, lr_schedule=None,
                 render=False, test_env=None, test_render=False,
                 test_episodes=1, test_maxsteps=5e5):
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
            test_episodes (int): Number of test episodes. To disable test evaluation, pass 0.
            test_maxsteps (int): Maximum step allowed during test per episode.
        """
        self.agent = agent
        self.maxsteps = maxsteps
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.logdir = logdir
        self.logfreq = logfreq
        self.log_on_term = log_on_term
        self.render = render
        self.test_env = agent.env if test_env is None else test_env
        self.test_render = test_render
        self.test_episodes = test_episodes
        self.test_maxsteps = test_maxsteps
        self._last_log_time = time.time()
        self.thread_env = thread_env
        if not isinstance(self.thread_env, CloudpickleWrapper):
            self.thread_env = CloudpickleWrapper(self.thread_env)
        self.thread_agents = thread_agents
        self.thread_stats = []
        # Make shared single counter
        for th_agent in thread_agents:
            th_agent.obs_step = agent.obs_step
            th_agent.train_step = agent.train_step
            th_agent.episode_step = agent.episode_step
            self.thread_stats.append(Stats(th_agent))

    @staticmethod
    def train_thread(thread_agent, stats, shared_net, env, request_stop, lr_schedule, batch_size):
        env = env()
        runner = EnvRunner(agent=thread_agent, env=env, batch_size=batch_size)
        while not request_stop.value:
            thread_agent.net.load_state_dict(shared_net.state_dict())
            rollout = runner.sample()
            thread_agent.train_on_batch(rollout,
                                        lr=lr_schedule.value(thread_agent.obs_step.value),
                                        summarize=False)
            stats.add(rollout)

    def train(self):
        """Starts training."""
        writer = None
        # writer = tf.summary.FileWriter(self.logdir, self.agent.sess.graph)
        request_stop = mp.Value('b', False)
        self.agent.net.share_memory()
        lr_schedule = Schedule.create(self.lr_schedule, torch_utils.get_lr(self.agent.opt),
                                      self.maxsteps)
        threads = []
        for thread_agent, stats in zip(self.thread_agents, self.thread_stats):
            t = mp.Process(target=AsyncTrainer.train_thread,
                           args=(thread_agent, stats, self.agent.net, self.thread_env,
                                 request_stop, lr_schedule, self.batch_size))
            t.daemon = True
            t.start()
            threads.append(t)
        last_log_time = time.time()
        try:
            while self.agent.obs_step.value < self.maxsteps:
                if time.time() - last_log_time >= self.logfreq:
                    last_log_time = time.time()
                    flush_stats(self.thread_stats, name="%s Thread" % self.agent.name,
                                maxsteps=self.maxsteps, writer=writer)
                    self.agent.save(self.logdir)
                    self.agent.test(self.test_env,
                                    self.test_episodes,
                                    maxsteps=self.test_maxsteps,
                                    render=self.test_render,
                                    writer=writer)
                if self.render:
                    [agent.env.render() for agent in self.thread_agents]
        except KeyboardInterrupt:
            logger.info('Caught Ctrl+C! Stopping training process.')
        request_stop.value = True
        logger.info('Saving progress & performing evaluation.')
        self.agent.save(self.logdir)
        self.agent.test(self.test_env, self.test_episodes, render=self.test_render)
        [t.join() for t in threads]
        logger.info('Training finished!')
        # writer.close()

    def save(self):
        pass

    def load(self):
        pass
