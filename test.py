import gym
import d4rl
import numpy as np
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from gym.wrappers import FlattenObservation

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


action = np.zeros((56,8,2))

def env_fn():
    return MultiStepWrapper(
        VideoRecordingWrapper(
            FlattenObservation(
                gym.make('maze2d-umaze-v1')
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=5,
                codec='h264',
                input_pix_fmt='rgb24',
                crf=22,
                thread_type='FRAME',
                thread_count=1
            ),
            file_path=None,
            steps_per_render=2
        ),
        n_obs_steps=8,
        n_action_steps=8,
        max_episode_steps=200
    )
env1 = gym.make('maze2d-umaze-v1')
env1.reset()
# env1.step(action)
env_fns = [env_fn] * 56
env = AsyncVectorEnv(env_fns)
# env.call_each('run_dill_function',
#                           args_list=[(x,) for x in this_init_fns])
env.step(action)
print("env reset: \n", env.reset())
# print(env.step(action))
