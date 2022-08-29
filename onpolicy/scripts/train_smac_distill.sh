#!/bin/sh
env="StarCraft2"
map="3s5z"
algo="distill"
exp="newKL"
seed_max=1
teacher_algo="MAT"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
     --teacher_algo ${teacher_algo} \
     --lr 5e-4 --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 2 --episode_length 100 --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval \
     --load_teacher '/home/eva_data/xyq/onpolicy/onpolicy/scripts/teacher.yml' --distill_epoch 10 --layer_N 2 # \
     # --model_dir /home/eva_data/xyq/onpolicy/onpolicy/scripts/results/StarCraft2/27m_vs_30m/rmappo/0811/wandb/run-20220811_101616-324qriz5/files # --use_wandb #--use_obs_instead_of_state True
    #  --add_center_xy --use_state_agent --gain 1 \
done