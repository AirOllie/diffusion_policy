import gym
import zarr
import numpy as np
# import d4rl # Import required to register environments, you may need to also import the submodule
import d4rl.kitchen

# Create the environment
env = gym.make('kitchen-complete-v0')

dataset = env.get_dataset()

# Open a Zarr group for writing
root_group = zarr.open_group('../data/kitchen/kitchen-complete-v0.zarr', mode='w')

# Create datasets within the group for each array in your dataset
# The dataset variable should contain your actual data
last_i = np.where(dataset['timeouts'])[0][-1] + 1
data_group = root_group.create_group('data')
data_group.create_dataset('action', data=dataset['actions'][:last_i], shape=dataset['actions'][:last_i].shape, dtype='float32')
data_group.create_dataset('obs', data=dataset['observations'][:last_i], shape=dataset['observations'][:last_i].shape, dtype='float32')

# Create a subgroup for metadata and add the episode ends dataset
meta_group = root_group.create_group('meta')
meta_group.create_dataset('episode_ends', data=np.where(dataset['timeouts'])[0],
                          shape=np.where(dataset['timeouts'])[0].shape, dtype='int64')

