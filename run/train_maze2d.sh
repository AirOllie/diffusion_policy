unset LD_PRELOAD
python train.py --config-dir=. --config-name=maze2d.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
