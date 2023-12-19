import gym
import zarr
import numpy as np
import d4rl # Import required to register environments, you may need to also import the submodule
import os

dataset_scale = 1.0

# Create the environment
env = gym.make('halfcheetah-expert-v0')

dataset = env.get_dataset()

# Determine the last index to include based on dataset_scale
episode_ends = np.where(dataset['timeouts'])[0] + 1
num_episodes = len(episode_ends)
episodes_to_include = int(num_episodes * dataset_scale)
last_i = episode_ends[episodes_to_include - 1]

# Open a Zarr group for writing
root_group = zarr.open_group('../data/halfcheetah/halfcheetah_expert_v0_{}.zarr'.format(dataset_scale), mode='w')

# Create datasets within the group for each array in your dataset
data_group = root_group.create_group('data')
data_group.create_dataset('action', data=dataset['actions'][:last_i], shape=dataset['actions'][:last_i].shape, dtype='float32')
data_group.create_dataset('obs', data=dataset['observations'][:last_i], shape=dataset['observations'][:last_i].shape, dtype='float32')

# Create a subgroup for metadata and add the episode ends dataset
meta_group = root_group.create_group('meta')
meta_group.create_dataset('episode_ends', data=episode_ends[:episodes_to_include], shape=episodes_to_include, dtype='int64')

print("Done!")