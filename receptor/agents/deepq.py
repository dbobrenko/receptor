from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import multiprocessing as mp
from torch.autograd import Variable

from receptor.agents.agent import BaseAgent
from receptor.core import losses
from receptor.core.policy import GreedyPolicy
from receptor.utils import torch_utils


class DeepQ(BaseAgent):
    def __init__(self, env, net, restore_from=None, use_double=False, trainable_net=None,
                 device='cuda:0', optimizer=None, policy=None, targetfreq=40000,
                 target_net=None, saver_keep=3, name='DeepQ'):
        """Constructs Deep Q-Learning agent.
         Includes the following implementations:
            1. Human-level control through deep reinforcement learning, Mnih et al., 2015.
            2. Dueling Network Architectures for Deep Reinforcement Learning, Wang et al., 2015.
                See `models.DeepQModel`.
            3. Deep Reinforcement Learning with Double Q-learning, Hasselt et al., 2016.
                See `use_double` argument.
            4. Prioritized Experience Replay, Schaul et al., 2015.
                See `core.replay.ProportionalReplay`.

        See `core.BaseDeepQ`.
        Args:
            env (gym.Env): Environment instance.
            net (models.AbstractFactory): Network factory.
            use_double (bool): Enables Double DQN.
            restore_from (str): Path to the pre-trained model.
            device (str): TensorFlow device, used for graph creation.
            optimizer (Optimizer or dict): Agent's optimizer.
                By default: RMSProp(lr=2.5e-4, momentum=0.95).
            targetfreq (int): Target network update frequency(in seen observations).
                Network architecture must be exactly the same as provided for this agent.
                If provided, current agent weights will remain constant.
                Pass None, to optimize current agent network.
            target_net (Network): Custom target network. Disables target sync.
                Pass None, to use agent's target network.
            saver_keep (int): Maximum number of checkpoints can be stored at once.
        """
        super(DeepQ, self).__init__(env=env, net=net, device=device,
                                    saver_keep=saver_keep, name=name)

        self.use_double = use_double
        self._target_freq = targetfreq
        self.policy = policy
        self.net = net
        self.last_sync = mp.Value('l', 0)

        self.target_net = target_net
        if target_net is None:
            import copy
            self.target_net = copy.deepcopy(self.net)
        self.net.to(device)
        self.target_net.to(device)
        self.target_net.last_sync = mp.Value('l', self.obs_step.value)
        self.target_net.eval()
        self.opt = optimizer
        self.trainable_net = net if trainable_net is None else trainable_net
        if self.opt is None:
            self.opt = torch.optim.RMSprop(self.trainable_net.parameters(),
                                           lr=0.00025, alpha=0.9, momentum=0.95, eps=0.01)
        elif isinstance(self.opt, dict):
            self.opt = torch_utils.create_optim(self.opt, self.trainable_net.parameters())
        if restore_from:
            self.load(restore_from)
        # tensor_utils.add_observation_summary(self.net['input'], self.env)
        # tf.summary.histogram('agent/action', self.actions)
        # tf.summary.histogram('agent/action_values', self.net['value'])
        # tf.summary.scalar('agent/learning_rate', self.opt.lr)
        # tf.summary.scalar('metrics/loss', loss)
        # tf.summary.scalar('metrics/avg_Q', tf.reduce_mean(q_next_max))
        # if restore_from and tf.train.latest_checkpoint(restore_from):
        #     self.load(restore_from)

    def predict_on_batch(self, obs_batch):
        """Computes action-values for given batch of observations."""
        obs_batch = Variable(torch.from_numpy(np.asarray(obs_batch, dtype='float32')))
        obs_batch = obs_batch.to(self.device)
        with torch.no_grad():
            return self.net(obs_batch)['Q']

    def act(self, obs):
        return self.act_on_batch([obs])

    def explore(self, obs):
        return self.explore_on_batch([obs])

    def act_on_batch(self, obs):
        """Computes action with greedy policy for given observation."""
        action_values = self.predict_on_batch(obs)
        return GreedyPolicy.select_action(self.env, action_values.data.cpu().numpy())

    def explore_on_batch(self, obs):
        action_values = self.predict_on_batch([obs])
        return self.policy.select_action(self.env, action_values.data.cpu().numpy(),
                                         step=self.obs_step)

    def load(self, checkpoint):
        super(DeepQ, self).load(checkpoint)
        self.sync_target_network()

    def sync_target_network(self):
        """Syncs target network with behaviour network."""
        self.target_net.load_state_dict(self.trainable_net.state_dict())

    def train_on_batch(self, rollout, lr=None, additional_losses=(), summarize=False, gamma=0.99):
        with self.train_step.get_lock():
            if self.train_step.value == self._target_freq:
                self.sync_target_network()
            self.train_step.value += 1
        with self.obs_step.get_lock():
            self.obs_step.value += len(rollout.obs)
        with self.episode_step.get_lock():
            self.episode_step.value += sum(rollout.terms)
        torch_utils.set_lr(self.opt, lr)
        rollout.compile()
        with torch.no_grad():
            if self.use_double:
                q_idx = self.net(rollout.obs_next)['Q'].max(1)[1]
                q_next = self.target_net(rollout.obs_next)['Q']
                q_next_max = torch.where(q_idx.to(torch.uint8), q_next[:, 1], q_next[:, 0])
            else:
                q_next_max = self.target_net(rollout.obs_next)['Q'].max(1)[0]
        rollout.discount_rewards(q_next_max.data.cpu().numpy(), gamma=gamma)
        self.net.zero_grad()
        self.trainable_net.zero_grad()
        rollout.outputs = self.net(rollout.obs)
        qloss = losses.QLoss(1.0)
        compositor = losses.LossCompositor([qloss])
        compositor.add(additional_losses)
        loss = compositor.loss(agent=self, rollouts=rollout)
        loss.backward()
        # torch_utils.clip_grads(self.net, -1, 1)
        torch_utils.copy_grads(from_net=self.net, to_net=self.trainable_net, device=self.device)
        self.opt.step()
        return rollout
