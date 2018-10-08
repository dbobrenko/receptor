from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable

from receptor.agents.agent import BaseAgent
from receptor.core import losses
from receptor.utils import torch_utils


class ActorCritic(BaseAgent):
    def __init__(self, env, net, restore_from=None, grad_norm=None, trainable_net=None, device='cuda:0',
                 optimizer=None, logdir=None, name='ActorCritic'):
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
            restore_from (str): Path to the pre-trained model.
            device (str): TensorFlow device, used for graph creation.
            optimizer (str or Optimizer): Agent's optimizer.
                By default: RMSProp(lr=2.5e-4, momentum=0.95).
                Network architecture must be exactly the same as provided for this agent.
                If provided, current agent weights will remain constant.
                Pass None, to optimize current agent network.
        """
        super(ActorCritic, self).__init__(env=env, net=net, device=device, logdir=logdir, name=name)
        self.net = net
        self.trainable_net = net if trainable_net is None else trainable_net
        self.opt = optimizer
        if self.opt is None:
            self.opt = torch.optim.RMSprop(self.trainable_net.parameters(),
                                           lr=7e-4, alpha=0.99, momentum=0.0, eps=0.1)
        elif isinstance(self.opt, dict):
            self.opt = torch_utils.create_optim(self.opt,
                                                self.trainable_net.parameters())
        self.net.to(self.device)
        if restore_from:
            self.load(restore_from)
        self.grad_norm = grad_norm

    def act(self, obs):
        return self.act_on_batch([obs])[0]

    def predict_on_batch(self, obs_batch):
        """Computes network output for given batch of observations."""
        obs_batch = Variable(torch.from_numpy(np.asarray(obs_batch, dtype='float32')))
        obs_batch = obs_batch.to(self.device)
        return self.net(obs_batch)

    def act_on_batch(self, obs_batch):
        """Computes action with maximum probability."""
        obs_batch = Variable(torch.from_numpy(np.asarray(obs_batch, dtype='float32')))
        obs_batch = obs_batch.to(self.device)
        return self.net.act(obs_batch)

    def explore_on_batch(self, obs_batch):
        """Computes action with given exploration policy for given observation."""
        obs_batch = Variable(torch.from_numpy(np.asarray(obs_batch, dtype='float32')))
        obs_batch = obs_batch.to(self.device)
        return self.net.explore(obs_batch)

    def train_on_batch(self, rollout, lr=None, additional_losses=(),
                       summarize=False, gamma=0.99):
        self.opt.zero_grad()
        torch_utils.set_lr(self.opt, lr)

        with torch.no_grad():
            actions, output = self.explore_on_batch(rollout.obs_next)
            expected_values = output['value'].cpu().numpy().squeeze(-1)
        rollout.discount_rewards(expected_values, gamma=0.99)
        if rollout.outputs is None:
            _, rollout.outputs = self.net.explore(rollout.obs)
        compositor = losses.LossCompositor()
        compositor.add(losses.PolicyGradientLoss(coef=1.0))
        compositor.add(losses.EntropyLoss(coef=-0.01))
        compositor.add(losses.TDLoss(coef=0.5))
        compositor.add(additional_losses)
        rollout.loss = compositor.loss(agent=self, rollouts=rollout)
        rollout.loss.backward()
        if self.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm)
        if self.net is not self.trainable_net:
            torch_utils.copy_grads(from_net=self.net, to_net=self.trainable_net, device=self.device)
        self.opt.step()

        with self.train_step.get_lock():
            self.train_step.value += 1
        with self.obs_step.get_lock():
            self.obs_step.value += len(rollout.obs)
        # with self.episode_step.get_lock():
        #     self.episode_step.value += torch.sum(rollout.terms)

        # if self.writer is not None:
        #     self.writer.add_scalar('Train/R', rewards)
        #     self.writer.add_scalar('Train/ObsStep', self.obs_step.value)
        #     self.writer.add_scalar('Train/TrainStep', self.train_step.value)
        #     self.writer.add_scalar('Train/EpisodeStep', self.episode_step.value)
        #     self.writer.add_scalar('Train/Loss', loss)
        #     self.writer.add_scalar('Train/Value', output['value'])
        #     self.writer.add_histogram('Train/Policy', output['policy'])
        #     torch_utils.add_observation_summary(obs, self.env)

        return rollout
