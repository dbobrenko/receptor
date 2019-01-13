from __future__ import absolute_import, print_function, division

import numpy as np
import numpy.testing as npt
from receptor.core.stats import RewardStats


def test_reward_stats_single_thread():
    stats = RewardStats()
    episodes = [[1, 0, 1, 1, 5, 10, 50],
                [0, 0, 0, 0, 0],
                [110, -1, 0, 0, -110],
                [-1]]
    avg_ep = np.mean([np.sum(ep) for ep in episodes])
    avg_step = np.sum([np.sum(ep) for ep in episodes]) / np.sum([len(ep) for ep in episodes])
    for ep in episodes:
        for i, r in enumerate(ep):
            stats.add(r, True if i == len(ep) - 1 else False)
    npt.assert_almost_equal(avg_ep, stats.episode_average(), 2)
    npt.assert_almost_equal(avg_step, stats.step_average(), 2)
    npt.assert_almost_equal(avg_ep, stats.reset_episode_rewards(), 2)
    npt.assert_almost_equal(avg_step, stats.reset_step_rewards(), 2)


def test_reward_stats_multi_thread():
    stats = RewardStats()
    avg_ep = (120 - 500 + 3) / 10
    avg_step = (120 - 500 + 3) / 20

    stats.add([0, 120, -100, 1], [False, False, True, False])
    stats.add([0, 0,   -100, 0], [False, False, True, False])
    stats.add([0, 0,   -100, 1], [True,  True,  True, False])
    stats.add([0, 0,   -100, 0], [True,  False, True, False])
    stats.add([0, 0,   -100, 1], [True,  False, True, True])
    npt.assert_almost_equal(avg_ep, stats.episode_average(), 2)
    npt.assert_almost_equal(avg_step, stats.step_average(), 2)
    npt.assert_almost_equal(avg_ep, stats.reset_episode_rewards(), 2)
    npt.assert_almost_equal(avg_step, stats.reset_step_rewards(), 2)
