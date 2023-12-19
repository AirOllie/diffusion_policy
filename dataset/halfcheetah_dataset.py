import gym
import zarr
import numpy as np
import d4rl # Import required to register environments, you may need to also import the submodule
import os

num_episodes = 200
traj_len = 100
env = gym.make('halfcheetah-expert-v0')

dataset = env.get_dataset()
episode_ends = np.arange(traj_len, num_episodes * traj_len + 1, traj_len)
last_i = num_episodes * traj_len

# Open a Zarr group for writing
root_group = zarr.open_group('../data/halfcheetah/halfcheetah_expert_data.zarr', mode='w')

# Create datasets within the group for each array in your dataset
data_group = root_group.create_group('data')
data_group.create_dataset('action', data=dataset['actions'][:last_i], shape=dataset['actions'][:last_i].shape, dtype='float32')
data_group.create_dataset('obs', data=dataset['observations'][:last_i], shape=dataset['observations'][:last_i].shape, dtype='float32')

# Create a subgroup for metadata and add the episode ends dataset
meta_group = root_group.create_group('meta')
meta_group.create_dataset('episode_ends', data=episode_ends, shape=episode_ends.shape, dtype='int64')

print("Done!")