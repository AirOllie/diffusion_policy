#!/bin/bash

# python train.py --config-dir=./config --config-name=halfcheetah_cnn_45.yaml hydra.run.dir="data/outputs/halfcheetah_cnn_45"
# python train.py --config-dir=./config --config-name=halfcheetah_cnn_45_chunk100.yaml hydra.run.dir="data/outputs/halfcheetah_cnn_45_chunk100"
# python train.py --config-dir=./config --config-name=halfcheetah_cnn_45_chunk300.yaml hydra.run.dir="data/outputs/halfcheetah_cnn_45_chunk300"

# python train.py --config-dir=./config --config-name=can_mh_cnn_45.yaml hydra.run.dir="data/outputs/can_mh_cnn_45"
# python train.py --config-dir=./config --config-name=can_mh_cnn_45.yaml hydra.run.dir="data/outputs/can_mh_cnn_45"
# python train.py --config-dir=./config --config-name=can_mh_cnn_45.yaml hydra.run.dir="data/outputs/can_mh_cnn_45"

# python train.py --config-dir=./config --config-name=toolhang_ph_cnn_45_20.yaml hydra.run.dir="data/outputs/toolhang_ph_cnn_45_20"
# python train.py --config-dir=./config --config-name=toolhang_ph_cnn_45_50.yaml hydra.run.dir="data/outputs/toolhang_ph_cnn_45_50"
# python train.py --config-dir=./config --config-name=toolhang_ph_cnn_45_100.yaml hydra.run.dir="data/outputs/toolhang_ph_cnn_45_100"

# python train.py --config-dir=./config --config-name=square_ph_cnn_45_20.yaml hydra.run.dir="data/outputs/square_ph_cnn_45_20"
# python train.py --config-dir=./config --config-name=square_ph_cnn_45_50.yaml hydra.run.dir="data/outputs/square_ph_cnn_45_50"
# python train.py --config-dir=./config --config-name=square_ph_cnn_45_100.yaml hydra.run.dir="data/outputs/square_ph_cnn_45_100"

#python train.py --config-dir=./config --config-name=can_ph_cnn.yaml hydra.run.dir="data/outputs/can_ph_cnn_45_20"
#python train.py --config-dir=./config --config-name=can_ph_cnn_45_50.yaml hydra.run.dir="data/outputs/can_ph_cnn_45_50"
#python train.py --config-dir=./config --config-name=can_ph_cnn_45_100.yaml hydra.run.dir="data/outputs/can_ph_cnn_45_100"

#python train.py --config-dir=./config --config-name=can_ph_cnn_45_20_0.1.yaml hydra.run.dir="data/outputs/can_ph_cnn_45_20_0.1"

# python train.py --config-dir=./config --config-name=square_ph_cnn_45_20_0.1.yaml hydra.run.dir="data/outputs/square_ph_cnn_45_20_0.1"

# ablation on diffusion steps
#python train.py --config-dir=./config --config-name=can_ph_cnn.yaml training.seed=45 \
#    task.dataset.dataset_scale=1.0 policy.noise_scheduler.num_train_timesteps=8 \
#    policy.num_inference_steps=8 hydra.run.dir="data/outputs/can_ph_cnn_45_8_1.0"
#python train.py --config-dir=./config --config-name=can_ph_cnn.yaml training.seed=45 \
#    task.dataset.dataset_scale=1.0 policy.noise_scheduler.num_train_timesteps=10 \
#    policy.num_inference_steps=10 hydra.run.dir="data/outputs/can_ph_cnn_45_10_1.0"
#python train.py --config-dir=./config --config-name=can_ph_cnn.yaml training.seed=45 \
#    task.dataset.dataset_scale=1.0 policy.noise_scheduler.num_train_timesteps=30 \
#    policy.num_inference_steps=30 hydra.run.dir="data/outputs/can_ph_cnn_45_30_1.0"

# ablation on dataset scale
#python train.py --config-dir=./config --config-name=can_ph_cnn.yaml training.seed=45 \
#    task.dataset.dataset_scale=0.3 policy.noise_scheduler.num_train_timesteps=8 \
#    policy.num_inference_steps=8 hydra.run.dir="data/outputs/can_ph_cnn_45_8_0.3"
python train.py --config-dir=./config --config-name=can_ph_cnn.yaml training.seed=45 \
    task.dataset.dataset_scale=0.5 policy.noise_scheduler.num_train_timesteps=8 \
    policy.num_inference_steps=8 hydra.run.dir="data/outputs/can_ph_cnn_45_8_0.5"

#python train.py --config-dir=./config --config-name=can_ph_cnn.yaml training.seed=45 \
#    task.dataset.dataset_scale=1.0 policy.noise_scheduler.num_train_timesteps=8 \
#    policy.num_inference_steps=8 hydra.run.dir="data/outputs/can_ph_cnn_45_8_1.0"