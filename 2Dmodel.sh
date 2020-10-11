env CUDA_VISIBLE_DEVICES=1 python train_F.py ./config/sep_anomal/mana_1.yaml --no_wandb > hoge1.out &
env CUDA_VISIBLE_DEVICES=1 python train_F.py ./config/sep_anomal/mana_2.yaml --no_wandb > hoge2.out &
env CUDA_VISIBLE_DEVICES=1 python train_F.py ./config/sep_anomal/mana_3.yaml --no_wandb > hoge3.out &
env CUDA_VISIBLE_DEVICES=1 python train_F.py ./config/sep_anomal/mana_4.yaml --no_wandb > hoge4.out &