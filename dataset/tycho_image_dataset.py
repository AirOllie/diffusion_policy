import csv
import zarr
import numpy as np
import PIL.Image as Image

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
        if int(row[1])!=traj_id:
            traj_id = int(row[1])
            episode_ends.append(int(row[0]))
    episode_ends.append(len(observations))

# # discard bad data
# observations = observations[episode_ends[4]:]
# actions = actions[episode_ends[4]:]
# episode_ends = np.array(episode_ends[5:]) - episode_ends[4]

observations = np.array(observations, dtype=np.float32)
eef_poses = observations[:, :3]
eef_quats = observations[:, 3:7]
gripper_poses = observations[:, 7]
actions = np.array(actions, dtype=np.float32)
episode_ends = np.array(episode_ends, dtype=np.int64)

# image data
images = []
for i in range(len(observations)):
    # load image with PIL
    img = Image.open('../data/tycho_reaching/imgs_kinect/kinect_{}.jpg'.format(i))
    img = img.resize((84, 84), Image.Resampling.LANCZOS)
    img = np.array(img, dtype=np.float32)
    images.append(img)

root_group = zarr.open_group('../data/tycho/tycho_reaching_image.zarr', mode='w')
data_group = root_group.create_group('data')

data_group.create_dataset('robot0_eef_pos', data=eef_poses, shape=eef_poses.shape, dtype='float32')
data_group.create_dataset('robot0_eef_quat', data=eef_quats, shape=eef_quats.shape, dtype='float32')
data_group.create_dataset('robot0_gripper_qpos', data=gripper_poses, shape=gripper_poses.shape, dtype='float32')
data_group.create_dataset('agentview_image', data=images, shape=(len(images), 84, 84, 3), dtype='uint8')
data_group.create_dataset('action', data=actions, shape=actions.shape, dtype='float32')

meta_group = root_group.create_group('meta')
meta_group.create_dataset('episode_ends', data=episode_ends, shape=episode_ends.shape, dtype='int64')

print('Done')