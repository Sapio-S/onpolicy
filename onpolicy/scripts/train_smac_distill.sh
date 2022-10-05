#!/bin/sh
env="StarCraft2"
map="27m_vs_30m"
algo="distill"
exp="rmappo_torchKL_sl_copy_generator_1004"
seed_max=1
teacher_algo="MAPPO"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
    --teacher_algo ${teacher_algo} \
    --ppo_epoch 5 --gain 0.01 --use_value_active_masks --use_eval \
    --n_rollout_threads 8 --episode_length 400 --num_env_steps 2500000 --num_mini_batch 1 \
    --load_teacher '/home/eva_data/xyq/onpolicy/onpolicy/scripts/teacher.yml' --distill_epoch 10 \
    --layer_N 2 #--model_dir '/home/eva_data/xyq/onpolicy/onpolicy/scripts/results/StarCraft2/3s5z/distill/rmappo_torchKL_sl_copy_rnn_1004/wandb/run-20221004_132708-12udcd1m/files'
    #--use_recurrent_policy #--use_wandb # --layer_N 2 
    
    #  --lr 5e-4 --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval \
    # --use_wandb # \
    #  --model_dir /home/eva_data/xyq/onpolicy/onpolicy/scripts/results/StarCraft2/3s5z/distill/dagger_mappo_ce/wandb/run-20220904_154456-355b3zua/files --use_wandb

    # --model_dir /home/eva_data/xyq/onpolicy/onpolicy/scripts/results/StarCraft2/27m_vs_30m/rmappo/0811/wandb/run-20220811_101616-324qriz5/files --use_wandb # --layer_N 2
    # --layer_N 2 --add_agent_id #  # \
    # --model_dir /home/eva_data/xyq/onpolicy/onpolicy/scripts/results/StarCraft2/27m_vs_30m/rmappo/0811/wandb/run-20220811_101616-324qriz5/files # --use_wandb #--use_obs_instead_of_state True
    #  --add_center_xy --use_state_agent --gain 1 \
done