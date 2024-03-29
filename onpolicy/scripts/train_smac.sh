

#!/bin/sh
env="StarCraft2"
map="6h_vs_8z"
exp="1011_coef_0.5"
seed_max=1

echo "env is ${env}, map is ${map}, exp is ${exp}, max seed is ${seed_max}"
for seed in {1..3..1}
do
    echo "seed is ${seed}:"

    algo="mappg"
    echo "algo is ${algo}"
    CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 8 --episode_length 400 --num_env_steps 10000000 --num_mini_batch 1 --ppo_epoch 5 --gain 0.01 --use_value_active_masks --use_eval --use_recurrent_policy

done
