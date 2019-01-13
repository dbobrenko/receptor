import numpy as np
import torch
from torch.autograd import Variable

from receptor.core.rollout import Rollout, RolloutParallel
from receptor.utils import discount_rewards


def test_rollouts():
    rollout = Rollout()
    obs = np.ones((2, 4, 3))
    for i in range(10):
        rollout.add(obs * i, action=5, reward=1, term=False)
    rollout.discount_rewards(1., gamma=0.5)
    expected = [1.99, 1.99, 1.99, 1.99, 1.98, 1.97, 1.94, 1.87, 1.75,  1.50]
    np.testing.assert_almost_equal(rollout.targets, expected, 2)


def test_rollouts_gamma0():
    # Discount test: scipy.signal.lfilter([1.], [1, -gamma], rewards[::-1])[::-1]
    rollouts = Rollout()
    gamma = 0.0
    ntrans = 10
    obs = [np.ones((2, 4, 3)) for _ in range(ntrans)]
    act = [5] * ntrans
    rew = [1.0] * ntrans
    term = [False] * ntrans
    term[-1] = True
    for o, a, r, t in zip(obs, act, rew, term):
        rollouts.add(o, action=a, reward=r, term=t)

    drew = [1.0] * ntrans
    rollouts.discount_rewards(1., gamma=gamma)
    np.testing.assert_array_almost_equal(rollouts.targets, drew, 3,
                                         err_msg="Incorrect discounted rewards w/ gamma=0.0"
                                                 " and terminal state.")


def test_rollouts_gamma09():
    # Trajectory #1
    rollouts = Rollout()
    gamma = 0.9
    ntrans = 5
    obs = [np.ones((5, 5, 1)) for _ in range(ntrans)]
    act = [5] * ntrans
    rew = [0.1, 0, 0, 5, 0]
    dr1 = [3.745, 4.05, 4.5, 5., 0.]
    term = [False, False, False, False, True]
    for o, a, r, t in zip(obs, act, rew, term):
        rollouts.add(o, action=a, reward=r, term=t)
    rollouts.discount_rewards(1., gamma=gamma)
    np.testing.assert_array_almost_equal(rollouts.targets, dr1, 3,
                                         err_msg="Incorrect discounted rewards for multiple"
                                                 "trajectory with len 5, gamma=0.9.")
    # Trajectory #2
    rollouts = Rollout()
    gamma = 0.5
    obs = np.ones((5, 5, 1))
    act = 5
    rew = 44
    dr2 = [44]
    rollouts.add(obs, action=act, reward=rew, term=[True])
    rollouts.discount_rewards([1.], gamma=gamma)
    np.testing.assert_array_almost_equal(rollouts.targets, dr2, 3,
                                         err_msg="Incorrect discounted rewards for multiple"
                                                 "trajectory with len 1, gamma=0.9.")

    # Trajectory #3
    rollouts = Rollout()
    gamma = 0.9
    ntrans = 8
    obs = [np.ones((5, 5, 1)) for _ in range(ntrans)]
    act = [1] * ntrans
    rew = [0.01, 1, 50, 0, 4.2, 0.5, 3, 0.02]
    dr3 = [46.0648, 51.17195, 55.74661, 6.38512, 7.09458, 3.2162, 3.018, 0.02]
    term = [False, False, False, False, False, False, False, True]
    for o, a, r, t in zip(obs, act, rew, term):
        rollouts.add(o, action=a, reward=r, term=t)
    rollouts.discount_rewards(1., gamma=gamma)
    np.testing.assert_array_almost_equal(rollouts.targets, dr3, 3,
                                         err_msg="Incorrect discounted rewards for multiple"
                                                 "trajectory with len 8, gamma=0.9.")


def test_parallel_rollouts():
    rollouts = RolloutParallel()
    gamma = 0.9
    ntrans = 6
    nproc = 4
    obs = [np.ones((5, 5, 1)) for _ in range(nproc) for _ in range(ntrans)]
    act = [[1] * nproc for _ in range(ntrans)]

    # Proc: P1   P2 P3  P4
    rew = [[50,   0, 1,  0],  # traj 1
           [0,    0, 2, -1],  # traj 2
           [4.2,  0, 3,  1],  # traj 3
           [0.5,  0, 4, -1],  # traj 4
           [3,    0, 5,  0],  # traj 5
           [0.02, 0, 6,  1],  # traj 6
           ]
    dr = [55.74661, 6.38512, 7.09458, 3.2162, 3.018, 0.02,
          0,              0,       0,      0,     0,    0,
          15.50088, 16.1121, 15.6801, 14.089, 11.21,  6.9,
          -0.22851, -0.2539,   0.829,  -0.19,   0.9,   1.
          ]
    term = [[False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [True,   True, False, True]
            ]
    ev = [1.] * nproc
    for o, a, r, t in zip(obs, act, rew, term):
        rollouts.add(o, action=a, reward=r, term=t)
    rollouts.discount_rewards(ev, gamma=gamma)
    np.testing.assert_array_almost_equal(rollouts.targets, dr, 3,
                                         err_msg="Incorrect discounted rewards for parallel"
                                                 "transitions w/ gamma=0.9.")
