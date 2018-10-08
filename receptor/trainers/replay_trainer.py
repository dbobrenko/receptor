from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from threading import Thread

import numpy as np

from receptor import logger
from receptor.core import ProportionalReplay, Stats, flush_stats
from receptor.core.runner import ReplayRunner
from receptor.core.schedule import Schedule
from receptor.trainers.trainer import BaseTrainer
from receptor.utils import torch_utils


class ReplayTrainer(BaseTrainer):
    def __init__(self, env, agent, maxsteps, replay,
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
            replay (core.ExperienceReplay): Experience replay buffer.
            render (bool): Enables game screen rendering.
            test_env (gym.Env): Environment instance, used for testing.
            test_render (bool): Enables rendering for test evaluations.
            test_episodes (int): Number of test episodes. To disable test evaluation, pass 0.
            test_maxsteps (int): Maximum step allowed during test per episode.
        """
        self.agent = agent
        self.maxsteps = maxsteps
        self.replay = replay
        self.logdir = logdir
        self.logfreq = logfreq
        self.log_on_term = log_on_term
        self.lr_schedule = lr_schedule
        self.render = render
        self.test_env = test_env
        self.test_render = test_render
        self.test_episodes = test_episodes
        self.test_maxsteps = test_maxsteps
        self.runner = ReplayRunner(agent=agent, env=env, replay=replay)
        self.train_stats = Stats(self.agent)
        self.perform_stats = Stats(self.agent)
        self._last_log_time = time.time()
        self._last_target_sync = self.agent.obs_step.value

    @staticmethod
    def collect_sample(obs, agent, replay, stats):
        action = agent.explore_on_batch([obs])
        obs_next, reward, term, info = agent.env.step(action)
        stats.add_rewards(rewards=reward, terms=term, info=info)
        replay.add(obs, action, reward, term, obs_next)
        obs = obs_next
        if term:
            obs = agent.env.reset()
        return obs

    @staticmethod
    def collect_replay(maxsteps, agent, replay, stats, render):
        obs = agent.env.reset()
        while agent.obs_step.value < maxsteps:
            if render:
                agent.env.render()
            obs = ReplayTrainer.collect_sample(obs, agent, replay, stats)
            time.sleep(0.01)

    def train(self):
        """Starts training."""
        try:
            lr_schedule = Schedule.create(self.lr_schedule, torch_utils.get_lr(self.agent.opt),
                                          self.maxsteps)
            writer = None
            # writer = tf.summary.FileWriter(self.logdir, self.agent.sess.graph)
            t = Thread(target=self.collect_replay, args=(self.maxsteps, self.agent, self.replay,
                                                         self.train_stats, self.render))

            t.daemon = True
            t.start()
            while self.agent.obs_step.value < self.maxsteps:
                if not self.replay.is_ready:
                    logger.info("Fulfilling minimum replay size %d/%d."
                                % (self.replay.size, self.replay.min_size))
                    time.sleep(2)
                    continue

                rollout = self.runner.sample()
                self.perform_stats.add(rollout)
                summarize = time.time() - self._last_log_time > self.logfreq
                lr = lr_schedule.value(self.agent.obs_step.value)
                self.agent.train_on_batch(rollout, lr=lr, summarize=summarize)

                if isinstance(self.replay, ProportionalReplay):
                    # TODO value methods
                    if 'value' in rollout.output:
                        prior = rollout.output['value'].squeeze(1).data.cpu().numpy()
                    elif 'Q' in rollout.output:
                        prior = np.abs(np.sum(rollout.output['Q'].data.cpu().numpy() * rollout.actions, 1))
                    else:
                        raise ValueError("Network must have `Q` or `value` output, "
                                         "in order to use ProportionalReplay")
                    self.replay.update(rollout.indexes, prior - rollout.target)

                if summarize:
                    self._last_log_time = time.time()
                    self.agent.save(self.logdir)
                    flush_stats(self.perform_stats, "%s Performance" % self.agent.name,
                                log_progress=False, log_rewards=False, log_hyperparams=False,
                                writer=writer)
                    flush_stats(self.train_stats, "%s Train" % self.agent.name,
                                log_performance=False, log_hyperparams=False,
                                maxsteps=self.maxsteps, writer=writer)
                    self.agent.test(self.test_env,
                                    self.test_episodes,
                                    maxsteps=self.test_maxsteps,
                                    render=self.test_render,
                                    writer=writer)
                    # if self.logdir and 'summary' in res:
                    #     writer.add_summary(res['summary'], global_step=self.agent.obs_step.value)
                    # writer.flush()

            logger.info('Performing final evaluation.')
            self.agent.test(self.test_env,
                            self.test_episodes,
                            maxsteps=self.test_maxsteps,
                            render=self.test_render)
            # writer.close()
            logger.info('Training finished.')
        except KeyboardInterrupt:
            logger.info('Stopping training process...')
        self.agent.save(self.logdir)

    def save(self):
        pass

    def load(self):
        pass
