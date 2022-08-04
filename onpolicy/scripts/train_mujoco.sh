#!/bin/sh
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

env="mujoco"
scenario="HalfCheetah-v2"
agent_conf="6x1"
agent_obsk=0
faulty_node=-1
#eval_faulty_node="-1 0 1 2 3 4 5"
eval_faulty_node="-1"
algo="rmappo"
exp="single"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=6 python train/train_mujoco.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --faulty_node ${faulty_node} --eval_faulty_node ${eval_faulty_node} \
--critic_lr 5e-3 --lr 5e-6 --entropy_coef 0.001 --max_grad_norm 0.5 --eval_episodes 5 --n_training_threads 16 --n_rollout_threads 40 --num_mini_batch 40 --episode_length 100 --eval_interval 25 --num_env_steps 10000000 \
--ppo_epoch 5 --clip_param 0.2 --layer_N 2 --use_eval --add_center_xy --use_state_agent --use_value_active_masks --use_policy_active_masks #--use_recurrent_policy False \
# --model_dir /home/eva_data/xyq/on-policy/onpolicy/scripts/results/mujoco/HalfCheetah-v2/mappo/single/wandb/run-20220706_181409-1hwpv0aq/files
# /home/eva_data/xyq/on-policy/onpolicy/scripts/results/mujoco/HalfCheetah-v2/mappo/single/wandb/run-20220706_181409-1hwpv0aq/files
# /home/eva_data/xyq/on-policy/onpolicy/scripts/results/mujoco/HalfCheetah-v2/mappo/single/wandb/run-20220705_102251-2kruuphg/files
# /home/eva_data/xyq/on-policy/onpolicy/scripts/results/mujoco/HalfCheetah-v2/mat/single/wandb/run-20220705_101414-19ze537y/files/transformer_2499.pt
# /home/eva_data/xyq/on-policy/onpolicy/scripts/results/mujoco/HalfCheetah-v2/mat_dec/single/wandb/run-20220705_101431-sl8tk6c5/files/transformer_2499.pt