#!/bin/sh
env="StarCraft2"
map="MMM2"
algo="distill"
exp="mat"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
     --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 2 --episode_length 100 --num_env_steps 1000000 --ppo_epoch 5 --use_value_active_masks --use_eval --add_center_xy --use_state_agent --gain 1 \
     --load_teacher '/home/eva_data/xyq/on-policy/onpolicy/scripts/teacher.yml' --distill_epoch 10 --use_wandb
    #--model_dir /home/eva_data/xyq/on-policy/onpolicy/scripts/results/StarCraft2/6h_vs_8z/mat/retrain/wandb/run-20220706_172156-3h1iua8u/files/transformer_3124.pt --use_wandb False -- use_obs_instead_of_state True
done