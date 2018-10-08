from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import torch

from receptor import logger


def create_optim(config, parameters):
    optim_map = {
        'adam': torch.optim.Adam,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD
    }
    if 'optim' not in config:
        raise ValueError("Config must contain field `optim` with optimizer name")
    if callable(config['optim']):
        optim = config['optim']
    elif isinstance(config['optim'], str):
        if config['optim'].lower() not in optim_map:
            raise ValueError("Unknown optimizer %s. Available: %s"
                             % (config['optim'], optim_map))
        optim = optim_map[config['optim'].lower()]
    del config['optim']
    return optim(parameters, **config)


def clip_grads(net, min_value, max_value):
    for param in net.parameters():
            param.grad.data.clamp_(min_value, max_value)


def set_lr(optim, lr):
    if lr is not None:
        for param_group in optim.param_groups:
            param_group['lr'] = lr


def get_lr(optim):
    return optim.param_groups[0]['lr']


def copy_grads(from_net, to_net, device='cuda:0'):
    if from_net is to_net:
        return
    for to_param, from_param in zip(to_net.parameters(),
                                    from_net.parameters()):
        # if to_param.grad is None:
        #     to_param.grad = from_param.grad.clone().to(device)
        # to_param.grad.data = from_param.grad.data.clone().to(device)
        if to_param.grad is not None:
            return
        to_param._grad = from_param.grad


# def discount_trajectory(rewards, terms, gamma, expected_values):
#     """Applies reward discounting for trajectories.
#
#     Args:
#         rewards (list): List of list of rewards for each trajectory.
#         terms (list): List of bools indicating terminal states for each trajectory.
#         gamma (float): Discount factor.
#         expected_values (list): List of expected future rewards for each trajectory.
#
#     Returns (list):
#         Discounted rewards.
#     """
#     for reward, term, ev in zip(rewards, terms, expected_values):
#         discount_sum = ev
#         for i in reversed(range(len(rewards))):
#             discount_sum = reward[i] + gamma * discount_sum
#             reward[i] = discount_sum
#     return rewards


def add_grads_summary(grad_vars):
    """Adds summary for weights and gradients.

    Args:
        grad_vars (list): List of (gradients, weights) tensors.
    """
    for grad, w in grad_vars:
        tf.summary.histogram(w.name, w)
        if grad is not None:
            tf.summary.histogram(w.name + '/gradients', grad)


def add_observation_summary(obs, env):
    """Adds observation summary.
    Supports observation tensors with 1, 2 and 3 dimensions only.
    1-D tensors logs as histogram summary.
    2-D and 3-D tensors logs as image summary.

    Args:
        obs (Tensor): Observation.
        env (gym.Env): Environment instance.
    """
    from receptor.envs.gym_wrap import ObservationStackWrap, ImageWrap
    # Get all wrappers
    all_wrappers = {}
    env_wrapper = env
    while True:
        if isinstance(env_wrapper, gym.Wrapper):
            all_wrappers[env_wrapper.__class__] = env_wrapper
            env_wrapper = env_wrapper.env
        else:
            break

    # Check for grayscale
    gray = False
    if ImageWrap in all_wrappers:
        gray = all_wrappers[ImageWrap].grayscale

    # Check and wrap observation stack
    if ObservationStackWrap in all_wrappers:
        channels = 1 if gray else 3
        for obs_id in range(all_wrappers[ObservationStackWrap].stack_len):
            o = obs[:, :, :, obs_id*channels:(obs_id+1)*channels]
            tf.summary.image('observation%d' % obs_id, o, max_outputs=1)
        return

    # Try to wrap current observation
    if len(env.observation_space.shape) == 1:
        tf.summary.histogram('observation', obs)
    elif len(env.observation_space.shape) == 2:
        tf.summary.image('observation', obs)
    elif len(env.observation_space.shape) == 3 and env.observation_space.shape[2] in (1, 3):
        tf.summary.image('observation', obs)
    else:
        logger.warn('Cannot create summary for observation with shape',
                    env.observation_space.shape)
