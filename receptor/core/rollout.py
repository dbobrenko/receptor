import numpy as np
import torch
from torch.autograd import Variable

from receptor.utils import discount_rewards, discount_rewards_onestep


class Rollout(object):
    def __init__(self, device='cuda:0'):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.obs_next = []
        self.terms = []
        self.trajends = []
        self.infos = []
        self.outputs = None
        self.targets = None
        self.loss = None
        self.device = device
        self._bootstrap_idx = None
        self.obs_bootstrap = None
        self.compiled = False

    def _to_tensor(self):
        self.obs = np.asarray(self.obs, dtype=np.float32)
        self.obs = np.concatenate(self.obs, 0)
        self.obs = Variable(torch.from_numpy(self.obs)).to(self.device)
        self.actions = np.asarray(self.actions, dtype=np.int32).flatten()
        self.actions = Variable(torch.from_numpy(self.actions)).to(self.device)
        self.terms = np.asarray(self.terms, dtype=np.int32)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)

    def add(self, obs, action, reward, term, output=None, info=None):
        """Adds transition to the rollout pool.

        Args:
            obs (list): Observations trajectory.
            action (list): Actions trajectory.
            reward (list): Rewards trajectory
            term (bool): If current trajectory ends up with terminal state.
            output (dict): Cached outputs (TODO).
            info (dict): Metadata (TODO).

        Returns:
            Flatten discounted rewards for every trajectory.
        """
        self.obs.append(np.copy(obs))
        self.actions.append(action)
        self.rewards.append(reward)
        self.terms.append(term)
        if info is not None:
            self.infos.append(info)

    def discount_rewards(self, expected_values, gamma=0.99):
        self._to_tensor()
        ev = np.asarray(expected_values, dtype=np.float32)
        if gamma > 0.0:
            targets = discount_rewards(self.rewards, gamma, ev, self.terms)
        else:
            targets = self.rewards.copy()
        self.targets = Variable(torch.from_numpy(targets)).to(self.device)
        return self.targets


class RolloutOneStep(Rollout):
    def __init__(self, device='cuda:0'):
        super(RolloutOneStep, self).__init__(device=device)
        self.importance = None
        self.indexes = None
        self.obs_next = None

    def discount_rewards(self, expected_values, gamma=0.99):
        self._to_tensor()
        ev = np.asarray(expected_values, dtype=np.float32)
        if gamma > 0.0:
            targets = discount_rewards_onestep(self.rewards, gamma, ev)
        else:
            targets = self.rewards.copy()
        self.targets = Variable(torch.from_numpy(targets)).to(self.device)
        return self.targets


class RolloutParallel(Rollout):
    def _to_tensor(self):
        self.obs = np.asarray(self.obs, dtype=np.float32).swapaxes(1, 0)
        self.obs = np.concatenate(self.obs, 0)
        self.obs = Variable(torch.from_numpy(self.obs)).to(self.device)
        self.actions = np.asarray(self.actions, dtype=np.int32).swapaxes(1, 0).flatten()
        self.actions = Variable(torch.from_numpy(self.actions)).to(self.device)
        self.terms = np.asarray(self.terms, dtype=np.int32).swapaxes(1, 0)
        self.rewards = np.asarray(self.rewards, dtype=np.float32).swapaxes(1, 0)

    def discount_rewards(self, expected_values, gamma=0.99):
        self._to_tensor()
        self.targets = self.rewards.copy()
        if gamma > 0.0:
            for n, (rew, term, value) in enumerate(zip(self.rewards, self.terms, expected_values)):
                value = value if term[-1] == 0 else 0.
                self.targets[n] = discount_rewards(rew, gamma, value, term)
            self.targets = np.asarray(self.targets, dtype=np.float32)
        targets = self.targets.flatten()
        self.targets = Variable(torch.from_numpy(targets)).to(self.device)
        return self.targets
