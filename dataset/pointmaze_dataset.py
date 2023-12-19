import gym
import zarr
import numpy as np
import minari
from minari import DataCollectorV0
import d4rl

dataset = minari.load_dataset("pointmaze-umaze-v1")

# Create the environment
# env  = dataset.recover_environment()


# Open a Zarr group for writing
root_group = zarr.open_group('../data/pointmaze/pointmaze.zarr', mode='w')

print(dataset)
last_i = np.where(dataset['timeouts'])[0][-1]
data_group = root_group.create_group('data')
data_group.create_dataset('action', data=dataset['actions'][:last_i], shape=dataset['actions'][:last_i].shape, dtype='float32')
data_group.create_dataset('obs', data=dataset['observations'][:last_i], shape=dataset['observations'][:last_i].shape, dtype='float32')

# Create a subgroup for metadata and add the episode ends dataset
meta_group = root_group.create_group('meta')
meta_group.create_dataset('episode_ends', data=np.where(dataset['timeouts'])[0],
                          shape=np.where(dataset['timeouts'])[0].shape, dtype='int64')


