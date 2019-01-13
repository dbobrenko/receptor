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
    def __init__(self, net, restore_from=None, grad_norm=None, trainable_net=None,
                 device='cuda:0', optimizer=None, logdir=None, name='ActorCritic'):
        """Actor Critic agent. See `core.BaseAgent`.
        Args:
            net (models.AbstractFactory): Network factory.
            restore_from (str): Path to the pre-trained model.
            device (str): TensorFlow device, used for graph creation.
            optimizer (str or Optimizer): Agent's optimizer.
                By default: RMSProp(lr=2.5e-4, momentum=0.95).
            trainable_net (
                Network architecture must be exactly the same as provided for this agent.
                If provided, current agent weights will remain constant.
                Pass None, to optimize current agent network.
        """
        super(ActorCritic, self).__init__(net=net, device=device, logdir=logdir, name=name)
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

    def explore(self, obs):
        return self.explore_on_batch([obs])[0]

    def predict_on_batch(self, obs_batch):
        obs_batch = Variable(torch.from_numpy(np.asarray(obs_batch, dtype='float32')))
        obs_batch = obs_batch.to(self.device)
        return self.net(obs_batch)

    def act_on_batch(self, obs_batch):
        obs_batch = Variable(torch.from_numpy(np.asarray(obs_batch, dtype='float32')))
        obs_batch = obs_batch.to(self.device)
        return self.net.act(obs_batch)

    def explore_on_batch(self, obs_batch):
        obs_batch = Variable(torch.from_numpy(np.asarray(obs_batch, dtype='float32')))
        obs_batch = obs_batch.to(self.device)
        return self.net.explore(obs_batch)

    def train_on_batch(self, rollout, lr=None, aux_losses=(), summarize=False, importance=None):
        self.opt.zero_grad()
        with self.train_step.get_lock():
            self.train_step.value += 1
        with self.obs_step.get_lock():
            self.obs_step.value += len(rollout.obs)
        # with self.episode_step.get_lock():
        #     self.episode_step.value += torch.sum(rollout.terms)
        torch_utils.set_lr(self.opt, lr)
        if importance is not None:
            importance = Variable(
                torch.from_numpy(np.asarray(importance, dtype='f'))).to(self.device)
        if rollout.outputs is None:
            _, rollout.outputs = self.net.explore(rollout.obs)
        compositor = losses.LossCompositor()
        compositor.add(losses.PolicyGradientLoss(coef=1.0))
        compositor.add(losses.EntropyLoss(coef=-0.01))
        compositor.add(losses.TDLoss(coef=0.5))
        compositor.add(aux_losses)
        loss = compositor.loss(agent=self, rollouts=rollout, importance=importance)
        loss.backward()
        rollout.loss = loss
        if self.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm)
        if self.net is not self.trainable_net:
            torch_utils.copy_grads(from_net=self.net, to_net=self.trainable_net, device=self.device)
        self.opt.step()
        return rollout
