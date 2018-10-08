from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class LossCompositor(object):
    def __init__(self, losses=None):
        self.losses = [] if losses is None else losses

    def add(self, loss):
        if isinstance(loss, (list, set, tuple)):
            self.losses += list(loss)
        else:
            self.losses.append(loss)

    def loss(self, agent, rollouts, **kwargs):
        value = 0
        for l in self.losses:
            value += l.loss(agent=agent, rollouts=rollouts, **kwargs)
        return value


class BaseLoss(object):
    def __init__(self, coef=1.0):
        self.coef = coef

    def loss(self, agent, rollouts):
        if not isinstance(rollouts.outputs, dict):
            raise ValueError("Network output must be a dict, got %s." % type(rollouts.outputs))


class PolicyGradientLoss(BaseLoss):
    def loss(self, agent, rollouts, **kwargs):
        """Policy Gradient loss.

        Args:
            output (dict): Dict with network endpoints.
                Must contain "policy" - action probabilities, "value" - baseline.
            action (Placeholder): Vectorized action placeholder.

        Returns (Tensor):
            Policy Gradient loss operation.
        """
        super(PolicyGradientLoss, self).loss(agent, rollouts)
        if 'dist' not in rollouts.outputs or 'value' not in rollouts.outputs:
            raise ValueError("Network output must contain dist and value fields,"
                             "in order to use %s. Got %s"
                             % (self.__class__.__name__, rollouts.outputs))
        logprob = -rollouts.outputs['dist'].log_prob(rollouts.actions)
        baseline = rollouts.targets - rollouts.outputs['value'].squeeze(-1)
        # pg = torch.sum(logprob * baseline.detach())
        pg = torch.mean(logprob * baseline.detach())
        return self.coef * pg


class EntropyLoss(BaseLoss):
    def __init__(self, coef=-0.01):
        super(EntropyLoss, self).__init__(coef=coef)

    def loss(self, agent, rollouts, **kwargs):
        if 'dist' not in rollouts.outputs:
            raise ValueError("Network output must contain dist field, in order to use %s. Got %s"
                             % (self.__class__.__name__, rollouts.outputs))
        return self.coef * rollouts.outputs['dist'].entropy().mean()
        # return self.coef * rollouts.outputs['dist'].entropy().sum()


class AdvantageLoss(BaseLoss):
    def loss(self, agent, rollouts, **kwargs):
        if 'value' not in rollouts.outputs:
            raise ValueError("Network output must contain value field, in order to use %s. Got %s"
                             % (self.__class__.__name__, rollouts.outputs))
        return self.coef * (rollouts.targets - rollouts.outputs['value'].squeeze(-1))


class QLoss(BaseLoss):
    """Action-Value Temporal-Difference error. See `td_error`."""

    def loss(self, agent, rollouts, **kwargs):
        if 'Q' not in rollouts.outputs:
            raise ValueError("Network output must contain Q field, in order to use %s. Got %s"
                             % (self.__class__.__name__, rollouts.outputs))
        td = rollouts.outputs['Q'][:, 0] * rollouts.actions.float() \
             + rollouts.outputs['Q'][:, 1] * (1 - rollouts.actions.float())
        importance = kwargs.get('importance', None)
        if importance is not None:
            td *= importance
        return self.coef * td.pow(2).mean()


class TDLoss(BaseLoss):
    def loss(self, agent, rollouts, **kwargs):
        if 'value' not in rollouts.outputs:
            raise ValueError("Network output must contain value field, in order to use %s. Got %s"
                             % (self.__class__.__name__, rollouts.outputs))
        td = rollouts.targets - rollouts.outputs['value'].squeeze(-1)
        importance = kwargs.get('importance', None)
        if importance is not None:
            td *= importance
        return self.coef * td.pow(2).mean()
