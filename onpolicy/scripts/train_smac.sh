#!/bin/sh
env="StarCraft2"
map="3s5z"
algo="mappo"
exp="3s5z_0729"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    # CUDA_VISIBLE_DEVICES=5 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 --lr 5e-4 --ppo_epoch 15 --clip_param 0.05 --save_interval 100000 --use_value_active_masks --use_eval --use_recurrent_policy False
    CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
    --ppo_epoch 5 --clip_param 0.2 --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 100 --num_env_steps 3000000 --ppo_epoch 15 --use_value_active_masks --use_eval --use_state_agent --lr 5e-4 --use_recurrent_policy False --layer_N 2
    #--use_stacked_frames True --stacked_frames 1 --use_recurrent_policy False # \
    #--model_dir /home/eva_data/xyq/on-policy/onpolicy/scripts/results/StarCraft2/6h_vs_8z/mat/retrain/wandb/run-20220706_172156-3h1iua8u/files/transformer_3124.pt --use_wandb False -- use_obs_instead_of_state True
done

# mappo: --ppo_epoch 5 --clip_param 0.2 \
# mat: --ppo_epoch 15 --clip_param 0.05 \

# mappo
#/home/eva_data/xyq/on-policy/onpolicy/scripts/results/StarCraft2/6h_vs_8z/mappo/retrain/wandb/run-20220706_180611-1qogpf0r
# mat
#/home/eva_data/xyq/on-policy/onpolicy/scripts/results/StarCraft2/6h_vs_8z/mat/retrain/wandb/run-20220706_172156-3h1iua8u/files/transformer_3124.pt
# mat_dec
#/home/eva_data/xyq/on-policy/onpolicy/scripts/results/StarCraft2/6h_vs_8z/mat_dec/retrain/wandb/run-20220706_172139-2edigroz/files/transformer_3124.pt