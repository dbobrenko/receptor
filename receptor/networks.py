from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from receptor.core import space


def _compute_conv_size(h, w, conv):
    h = ((h - conv.kernel_size[0] + 2 * conv.padding[0])
         / conv.stride[0]) + 1
    w = ((w - conv.kernel_size[1] + 2 * conv.padding[1])
         / conv.stride[1]) + 1
    return h, w, conv.out_channels


class FullyConnected(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers=(256, 256)):
        super(FullyConnected, self).__init__()
        self.layers = []
        self.inputs = nn.Linear(input_shape, hidden_layers[0])
        self.hidden = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.value = nn.Linear(hidden_layers[1], output_shape)

    @classmethod
    def from_env(cls, env, hidden_layers=(256, 256)):
        input_shape = env.observation_space.shape
        output_shape = env.action_space.shape
        if len(input_shape) > 1:
            raise ValueError("Current environment has multiple inputs (%s). "
                             "Please consider providing custom network." % input_shape.ndim)
        if len(output_shape) > 1:
            raise ValueError("Current environment has multiple outputs (%s). "
                             "Please consider providing custom network." % output_shape.ndim)
        return cls(input_shape[0], output_shape[0], hidden_layers=hidden_layers)

    def forward(self, x):
        activations = {}
        activations['fc1'] = self.inputs(x)
        activations['relu1'] = F.relu(activations['fc1'])
        activations['fc2'] = self.hidden(activations['relu1'])
        activations['relu2'] = F.relu(activations['fc2'])
        activations['Q'] = self.value(activations['relu2'])
        return activations


class DeepQNetwork(nn.Module):
    """Deep Q-Network model as defined in:
    Human-level control through deep reinforcement learning, Mnih et al., 2015.
    Args:
        nature_arch (bool): If enabled, uses architecture as defined in Mnih et al., 2015,
            otherwise in Mnih et al., 2013.
        dueling (bool): If enabled, uses dueling head as defined in Wang et al., 2015.
    """
    def __init__(self, input_shape, output_shape, nature_arch=True,
                 dueling=False, nwhc_input=True):
        super(DeepQNetwork, self).__init__()
        self.nature_arch = nature_arch
        self.dueling = dueling
        self.nwhc_input = nwhc_input
        if self.nwhc_input:
            input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.conv_layers = []
        self.conv1 = nn.Conv2d(input_shape[0],
                               32 if self.nature_arch else 16,
                               kernel_size=8,
                               stride=4,
                               padding=0,
                               bias=True)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        h, w, c = _compute_conv_size(input_shape[1], input_shape[2], self.conv1)
        self.conv_layers.append(self.conv1)
        self.conv2 = nn.Conv2d(32 if self.nature_arch else 16,
                               64 if self.nature_arch else 32,
                               kernel_size=4,
                               stride=2,
                               padding=0,
                               bias=True)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        h, w, c = _compute_conv_size(h, w, self.conv2)
        self.conv_layers.append(self.conv2)
        if self.nature_arch:
            self.conv3 = nn.Conv2d(64,
                                   64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=0,
                                   bias=True)
            torch.nn.init.xavier_uniform_(self.conv3.weight)
            h, w, c = _compute_conv_size(h, w, self.conv3)
            self.conv_layers.append(self.conv3)

        if self.dueling:
            raise NotImplementedError
        else:
            self.fc = nn.Linear(int(h * w * c), 512 if self.nature_arch else 256)
            self.output = nn.Linear(512 if self.nature_arch else 256, output_shape)

    def forward(self, x):
        if self.nwhc_input:
            x = x.permute(0, 3, 1, 2)
        activations = {}
        activations['conv1'] = F.relu(self.conv1(x))
        x = F.relu(self.conv2(activations['conv1']))
        activations['conv2'] = x
        if self.nature_arch:
            x = F.relu(self.conv3(x))
            activations['conv3'] = x
        activations['fc1'] = F.relu(self.fc(x.view(x.size(0), -1)))
        activations['Q'] = self.output(activations['fc1'])
        return activations['Q'], activations


class ActorCriticConv(nn.Module):
    def __init__(self, input_shape, output_shape, nature_arch=True,
                 nwhc_input=True, continuous=False):
        super(ActorCriticConv, self).__init__()
        self.nature_arch = nature_arch
        self.nwhc_input = nwhc_input
        self.continuous = continuous
        if self.nwhc_input:
            input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.conv_layers = []
        self.conv1 = nn.Conv2d(input_shape[0],
                               32 if self.nature_arch else 16,
                               kernel_size=8,
                               stride=4,
                               padding=0,
                               bias=True)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        h, w, c = _compute_conv_size(input_shape[1], input_shape[2], self.conv1)
        self.conv_layers.append(self.conv1)
        self.conv2 = nn.Conv2d(32 if self.nature_arch else 16,
                               64 if self.nature_arch else 32,
                               kernel_size=4,
                               stride=2,
                               padding=0,
                               bias=True)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        h, w, c = _compute_conv_size(h, w, self.conv2)
        self.conv_layers.append(self.conv2)
        if self.nature_arch:
            self.conv3 = nn.Conv2d(64,
                                   64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=0,
                                   bias=True)
            torch.nn.init.xavier_uniform_(self.conv3.weight)
            h, w, c = _compute_conv_size(h, w, self.conv3)
            self.conv_layers.append(self.conv3)

        self.fc = nn.Linear(int(h * w * c), 512 if self.nature_arch else 256)
        self.value = nn.Linear(512 if self.nature_arch else 256, 1)
        if self.continuous:
            self.mean = nn.Linear(512 if self.nature_arch else 256, output_shape)
            self.std = nn.Linear(512 if self.nature_arch else 256, output_shape)
        else:
            self.policy = nn.Linear(512 if self.nature_arch else 256, output_shape)

    def forward(self, x):
        if self.nwhc_input:
            x = x.permute(0, 3, 1, 2)
        activations = {}
        x = F.relu(self.conv1(x))
        activations['conv1'] = x
        x = F.relu(self.conv2(x))
        activations['conv2'] = x
        if self.nature_arch:
            x = F.relu(self.conv3(x))
            activations['conv3'] = x
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        activations['fc1'] = x
        value = self.value(x)
        activations['value'] = value
        if self.continuous:
            mean = F.tanh(self.mean(x))
            std = F.softplus(self.std(x))
            activations['policy'] = torch.distributions.Normal(mean, std)
        else:
            activations['policy_logits'] = self.policy(x)
            probs = F.softmax(activations['policy_logits'], dim=-1)
            activations['dist'] = torch.distributions.Categorical(probs)
        return activations['dist'], activations

    def explore(self, x):
        pd, activations = self.forward(x)
        activations['policy'] = pd.sample()
        return activations['policy'].cpu().numpy().squeeze(), activations

    def act(self, x):
        pd, activations = self.forward(x)
        if self.continuous:
            activations['policy'] = pd.mean.cpu().numpy().squeeze()
        else:
            activations['policy'] = torch.max(pd.probs, 1)[1]
        return activations['policy'].cpu().numpy().squeeze(), activations


class ActorCriticFC(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers=(256, 256), continuous=False):
        super(ActorCriticFC, self).__init__()
        self.continuous = continuous
        self.inputs = nn.Linear(input_shape, hidden_layers[0])
        self.hidden = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.value = nn.Linear(hidden_layers[1], 1)
        if self.continuous:
            self.mean = nn.Linear(hidden_layers[1], output_shape)
            self.std = nn.Linear(hidden_layers[1], output_shape)
        else:
            self.policy = nn.Linear(hidden_layers[1], output_shape)

    @classmethod
    def from_env(cls, env, hidden_layers=(256, 256)):
        input_shape = env.observation_space.shape
        output_shape = env.action_space.shape
        if len(input_shape) > 1:
            raise ValueError("Current environment has multiple inputs (%s). "
                             "Please consider providing custom network." % input_shape.ndim)
        if len(output_shape) > 1:
            raise ValueError("Current environment has multiple outputs (%s). "
                             "Please consider providing custom network." % output_shape.ndim)
        return cls(input_shape[0], output_shape[0], hidden_layers=hidden_layers,
                   continuous=isinstance(env.action_space, space.Continuous))

    def forward(self, x):
        activations = {}
        activations['fc1'] = self.inputs(x)
        activations['relu1'] = F.relu(activations['fc1'])
        activations['fc2'] = self.hidden(activations['relu1'])
        activations['relu2'] = F.relu(activations['fc2'])
        activations['value'] = self.value(activations['relu2'])
        if self.continuous:
            mean = F.tanh(self.mean(activations['relu2']))
            std = F.softplus(self.std(activations['relu2']))
            activations['policy'] = torch.distributions.Normal(mean, std)
        else:
            activations['policy_logits'] = self.policy(activations['relu2'])
            probs = F.softmax(activations['policy_logits'], dim=-1)
            activations['dist'] = torch.distributions.Categorical(probs)
        return activations['dist'], activations

    def explore(self, x):
        pd, activations = self.forward(x)
        return pd.sample().cpu().numpy().squeeze(), activations

    def act(self, x):
        pd, activations = self.forward(x)
        if self.continuous:
            return pd.mean.cpu().numpy(), activations
        return torch.max(pd.probs, 1)[1].cpu().numpy().squeeze(), activations

# def dueling_header(input_layer, output_shape, dueling_type='mean',
#                    advantage_layers=(512,), value_layers=(512,), trainable=True):
#     layers = []
#     for i, units in enumerate(advantage_layers):
#         layers.append(nn.Linear(units, output_shape))
#     adv_layer = tf.layers.dense(adv_layer, units=output_space.shape[0],
#                                 activation=None, name='adv_out', trainable=trainable)
#
#     value_layer = input_layer
#     for i, units in enumerate(value_layers):
#         value_layer = tf.layers.dense(value_layer, units=units,
#                                       activation=tf.nn.relu, trainable=trainable,
#                                       name='value%d' % i)
#     value_layer = tf.layers.dense(value_layer, units=1, activation=None,
#                                   name='value_out', trainable=trainable)
#     if dueling_type == 'naive':
#         out = value_layer + adv_layer
#     elif dueling_type == 'mean':
#         out = value_layer + (adv_layer - tf.reduce_mean(adv_layer, 1, keepdims=True))
#     elif dueling_type == 'max':
#         out = value_layer + (adv_layer - tf.reduce_max(adv_layer, 1, keepdims=True))
#     else:
#         raise ValueError("Unknown dueling type '%s'. Available: 'naive', 'mean', 'max'."
#                          % dueling_type)
#     return out
