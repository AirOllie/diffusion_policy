import csv
import zarr
import numpy as np
from scipy.interpolate import interp1d

# Read data
observations = []
actions = []
episode_ends = []
with open('../data/tycho_reaching/label_kinect.csv', 'r') as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # skip header
    traj_id = 0
    for row in csv_reader:
        observations.append(row[4].strip('[]').split())  # row['choppose']
        actions.append(row[5].strip('[]').split())  # row['choppose_target']
        if int(row[1]) != traj_id:
            traj_id = int(row[1])
            episode_ends.append(int(row[0]))
    episode_ends.append(len(observations))

# discard bad data
# observations = observations[episode_ends[4]:]
# actions = actions[episode_ends[4]:]
# episode_ends = np.array(episode_ends[5:]) - episode_ends[4]


# Convert to numpy arrays
observations = np.array(observations, dtype=np.float32)
actions = np.array(actions, dtype=np.float32)


# Function to interpolate
def interpolate_data(data, episode_ends):
    new_data = []
    new_episode_ends = []
    current_index = 0
    for end in episode_ends:
        episode_data = data[current_index:end]
        x = np.arange(len(episode_data))
        x_new = np.linspace(0, len(episode_data) - 1, len(episode_data) * 4 - 3)
        interpolator = interp1d(x, episode_data, axis=0, kind='linear')
        new_episode_data = interpolator(x_new)

        new_data.append(new_episode_data)
        current_index = end
        new_episode_ends.append(sum([len(d) for d in new_data]))

    return np.concatenate(new_data), np.array(new_episode_ends, dtype=np.int64)


# Interpolate observations and actions
interpolated_observations, new_episode_ends = interpolate_data(observations, episode_ends)
interpolated_actions, _ = interpolate_data(actions, episode_ends)

# Store in Zarr format
root_group = zarr.open_group('../data/tycho/tycho_reaching_interpolated.zarr', mode='w')
data_group = root_group.create_group('data')

data_group.create_dataset('obs', data=interpolated_observations, shape=interpolated_observations.shape, dtype='float32')
data_group.create_dataset('action', data=interpolated_actions, shape=interpolated_actions.shape, dtype='float32')

meta_group = root_group.create_group('meta')
meta_group.create_dataset('episode_ends', data=new_episode_ends, shape=new_episode_ends.shape, dtype='int64')

print('Done')

# plot each dimension of obs and action to see if they are interpolated correctly
import matplotlib.pyplot as plt
plt.figure()
for i in range(8):
    plt.subplot(4,2,i+1)
    # stretch observations three times
    plt.plot(np.repeat(observations[:,i],4))
    plt.plot(interpolated_observations[:,i])
    # plot episode ends
    for end in new_episode_ends:
        plt.axvline(x=end, color='r')
plt.savefig('obs.png')

plt.figure()
for i in range(8):
    plt.subplot(4,2,i+1)
    # stretch actions three times
    plt.plot(np.repeat(actions[:,i],3))
    plt.plot(interpolated_actions[:,i])
    for end in new_episode_ends:
        plt.axvline(x=end, color='r')
plt.savefig('action.png')