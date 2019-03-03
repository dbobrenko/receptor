from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import receptor
except ImportError:
    import os.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import receptor
from receptor.agents import ActorCritic
from receptor.envs.gym_wrap import GymWrap
from receptor.networks import ActorCriticFC

from receptor.envs.parallel_envs import AsyncEnvs
from receptor.trainers import SyncTrainer
import gym

receptor.set_random_seed(555)

env_name = 'CartPole-v0'
make_env = lambda: GymWrap(gym.make(env_name))

agent = ActorCritic(net=ActorCriticFC(4, 2),
                    optimizer={'optim': 'adam', 'lr': 0.0005},
                    grad_norm=0.5
                    )

envs = AsyncEnvs([make_env, make_env, make_env, make_env])

trainer = SyncTrainer(agent,
                      envs,
                      test_env=make_env(),
                      lr_schedule="linear",
                      maxsteps=500000,
                      batch_size=20,
                      logdir='/tmp/prl/A2C/%s' % env_name,
                      logfreq=10
                      )
trainer.train()
