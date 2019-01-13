from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import numpy.testing as npt
from receptor.core import ExperienceReplay, ProportionalReplay


def test_replay_add():
    cap = 10000
    replay = ExperienceReplay(capacity=cap, min_size=500, batch_size=32)
    for i in range(3*cap):
        replay.add(0, 0, 0, 0, False)
    assert replay.size == cap


def test_replay_sample():
    cap = 256
    batch_size = 32
    replay = ExperienceReplay(capacity=cap, min_size=cap, batch_size=batch_size)
    for i in range(0, 10*cap, 10):
        replay.add(obs=i, action=i+1, reward=i+2, obs_next=i+10, term=i % 20)
    for _ in range(10):
        rollout = replay.sample()
        for o, a, r, o_next, t in zip(rollout.obs, rollout.actions, rollout.rewards,
                                      rollout.obs_next, rollout.terms):
            assert a - 1 == o
            assert r - 2 == o
            if not t:
                assert o_next - 10 == o
            assert t == o % 20
        assert len(rollout.obs) == batch_size


def test_replay_sample_term():
    cap = 1024
    batch_size = 128
    replay = ExperienceReplay(capacity=cap, min_size=cap, batch_size=batch_size)
    for i in range(cap):
        replay.add(obs=i, action=0, reward=0, obs_next=i+1, term=i % 5 == 0)
    rollout = replay.sample()
    assert len(rollout.obs) == batch_size
    assert len(rollout.obs_next) == batch_size
    assert len(rollout.actions) == batch_size
    assert len(rollout.rewards) == batch_size
    assert len(rollout.terms) == batch_size
    for o, o_next, term in zip(rollout.obs, rollout.obs_next, rollout.terms):
        if not term:
            assert o+1 == o_next


def test_proportional_add():
    cap = 10000
    replay = ProportionalReplay(capacity=cap, min_size=500, batch_size=32, alpha=1, beta=1)
    for i in range(3*cap):
        replay.add(0, 0, 0, 0, False)
    assert replay.size == cap


def test_prop_replay_sample():
    cap = 512
    batch_size = 256
    replay = ProportionalReplay(capacity=cap, min_size=cap, batch_size=batch_size, alpha=1, beta=1)
    for i in range(2*cap):
        replay.add(obs=i, action=0, reward=0, obs_next=i+1, term=False, priority=i)
    rollout = replay.sample()
    assert len(rollout.obs) == batch_size
    assert len(rollout.obs_next) == batch_size
    assert len(rollout.actions) == batch_size
    assert len(rollout.rewards) == batch_size
    assert len(rollout.terms) == batch_size
    npt.assert_equal([i+1 for i in rollout.obs],
                     rollout.obs_next)


def test_prop_replay_distribution():
    priors = [20000.0, 30000.0, 1000.0, 49000.0, 0.0]
    cap = 256
    batch_size = 32
    sample_amount = 2000
    replay = ProportionalReplay(capacity=cap, min_size=cap, batch_size=batch_size, alpha=1, beta=1)
    s = int(np.sum(priors))
    expected_priors = np.asarray(priors) / s
    received_priors = [0] * len(priors)
    for o, p in enumerate(priors):
        replay.add(obs=o, action=0, reward=0, obs_next=0, term=False, priority=p)
    for i in range(sample_amount):
        rollout = replay.sample()
        for o in rollout.obs:
            received_priors[o] += 1
    received_priors = np.asarray(received_priors) / (sample_amount*batch_size)
    npt.assert_almost_equal(expected_priors, received_priors, decimal=2)


def test_prop_replay_update():
    priors = np.array([20000.0, 30000.0, 1000.0, 49000.0, 0.0])
    cap = 2048
    batch_size = 32
    sample_amount = 2000
    replay = ProportionalReplay(capacity=cap, min_size=cap, batch_size=batch_size, alpha=1, beta=1)
    s = int(np.sum(priors))
    expected_priors = priors / s
    received_priors = [0] * len(priors)
    for o, p in enumerate(priors):
        replay.add(obs=o, action=0, reward=0, obs_next=0, term=False)
    replay.update(list(range(len(priors))), priors)
    for i in range(sample_amount):
        rollout = replay.sample()
        for o in rollout.obs:
            received_priors[o] += 1
    received_priors = np.asarray(received_priors) / (sample_amount*batch_size)
    npt.assert_almost_equal(expected_priors, received_priors, decimal=2)
