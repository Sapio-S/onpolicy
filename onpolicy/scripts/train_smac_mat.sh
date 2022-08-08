#!/bin/sh
env="StarCraft2"
map="3s5z_vs_3s6z"
algo="mat"
exp="0808"
seed_max=5

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
     --use_recurrent_policy False \
    --use_value_active_masks --use_eval --max_grad_norm 10  --layer_N 1 \
    --ppo_epoch 5 --clip_param 0.2 --use_stacked_frames True --stacked_frames 1 \
    --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 --lr 5e-4
done