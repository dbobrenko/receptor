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
from receptor.networks import ActorCriticConv
from receptor.envs import AtariWrapper

from receptor.envs.parallel_envs import AsyncEnvs, ParallelFrameStack
from receptor.trainers import SyncTrainer

receptor.set_random_seed(555)


env_name = "BreakoutDeterministic-v4"
num_act = 4
num_stack = 4


def make_atari(train=True):
    return lambda: AtariWrapper(env_name,
                                obs_stack=1 if train else num_stack,
                                new_width=84,
                                new_height=84,
                                to_gray=True,
                                start_action=1,
                                noop_action=0 if train else None,
                                clip_rewards=train)


agent = ActorCritic(net=ActorCriticConv(input_shape=(84, 84, num_stack), output_shape=num_act),
                    optimizer={"optim": "rmsprop", "lr": 7e-4, "alpha": 0.99, "eps": 1e-5},
                    grad_norm=0.5)
envs = []
for i in range(16):
    envs.append(make_atari())

envs = AsyncEnvs(envs)
envs = ParallelFrameStack(envs, num_stack)

trainer = SyncTrainer(agent,
                      envs,
                      lr_schedule="linear",
                      maxsteps=80000000,
                      batch_size=5,
                      logdir='/tmp/receptor/A2C/%s' % env_name,
                      logfreq=120,
                      test_env=make_atari(train=False)(),
                      render=False,
                      test_render=False
                      )
trainer.train()
