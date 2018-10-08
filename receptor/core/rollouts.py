import numpy as np
import torch
from torch.autograd import Variable

from receptor.utils import discount_rewards, discount_rewards_onestep


class Rollouts(object):
    def __init__(self, device='cuda:0'):
        self.obs = None
        self._obs = []
        self.actions = None
        self._actions = []
        self.rewards = None
        self._rewards = []
        self.terms = None
        self._terms = []
        self.infos = []
        self._obs_next = None
        self.outputs = None
        self.targets = None
        self.loss = None
        self.device = device

    @property
    def obs_next(self):
        return self._obs_next

    @obs_next.setter
    def obs_next(self, obs_next):
        self._obs_next = np.asarray(obs_next, dtype=np.float32)
        self._obs_next = Variable(torch.from_numpy(self.obs_next)).to(self.device)

    def _update_tensors(self):
        self.obs = np.asarray(self._obs, dtype=np.float32)
        self.obs = Variable(torch.from_numpy(self.obs)).to(self.device)
        self.obs_next = np.asarray(self._obs_next, dtype=np.float32)
        self.obs_next = Variable(torch.from_numpy(self.obs_next)).to(self.device)
        self.actions = np.asarray(self._actions, dtype=np.int32)
        self.actions = Variable(torch.from_numpy(self.actions)).to(self.device)
        self.terms = np.asarray(self._terms, dtype=np.int32)
        self.rewards = np.asarray(self._rewards, dtype=np.float32)

    def add(self, obs, action, reward, term, info=None):
        """Adds transition to the rollout pool.

        Args:
            obs (list): Observations trajectory.
            action (list): Actions trajectory.
            reward (list): Rewards trajectory
            term (bool): If current trajectory ends up with terminal state.
            info (dict): Metadata (TODO).

        Returns:
            Flatten discounted rewards for every trajectory.
        """
        self._obs.append(np.copy(obs))
        self._actions.append(action)
        self._rewards.append(reward)
        self._terms.append(term)
        if info is not None:
            self.infos.append(info)

    def discount_rewards(self, expected_values, gamma=0.99):
        self._update_tensors()
        ev = np.asarray(expected_values, dtype=np.float32)
        if gamma > 0.0:
            self.targets = discount_rewards(self.rewards, gamma, ev, self.terms)
        else:
            self.targets = self.rewards.copy()
        return self.targets


class RolloutsOneStep(Rollouts):
    def __init__(self, device='cuda:0'):
        super(RolloutsOneStep, self).__init__(device=device)
        self.importance = None
        self.indexes = None
        self.obs_next = None

    def discount_rewards(self, expected_values, gamma=0.99):
        self._update_tensors()
        ev = np.asarray(expected_values, dtype=np.float32)
        if gamma > 0.0:
            targets = discount_rewards_onestep(self.rewards, gamma, ev)
        else:
            targets = self.rewards.copy()
        self.targets = Variable(torch.from_numpy(targets)).to(self.device)
        return self.targets


class ParallelRollouts(Rollouts):
    def _update_tensors(self):
        self.obs = np.asarray(self._obs, dtype=np.float32).swapaxes(1, 0)
        self.obs = np.concatenate(self.obs, 0)
        self.obs = Variable(torch.from_numpy(self.obs)).to(self.device)
        self.actions = np.asarray(self._actions, dtype=np.int32).swapaxes(1, 0).flatten()
        self.actions = Variable(torch.from_numpy(self.actions)).to(self.device)
        self.terms = np.asarray(self._terms, dtype=np.int32).swapaxes(1, 0)
        self.rewards = np.asarray(self._rewards, dtype=np.float32).swapaxes(1, 0)
        self.targets = self.rewards.copy()

    def discount_rewards(self, ev, gamma=0.99):
        self._update_tensors()
        if gamma > 0.0:
            for n, (rewards, dones, value) in enumerate(zip(self.rewards, self.terms, ev)):
                value = value if dones[-1] == 0 else 0.
                self.targets[n] = discount_rewards(rewards, gamma, value, dones)
        self.targets = self.targets.flatten().astype("float32")
        self.targets = Variable(torch.tensor(self.targets)).to(self.device)
        return self.targets
