# #!/bin/sh
# env="StarCraft2"
# map="3s5z"
# exp="0826"
# seed_max=1

# echo "env is ${env}, map is ${map}, exp is ${exp}, max seed is ${seed_max}"
# for seed in {3..3..1}
# do
#     echo "seed is ${seed}:"

#     algo="mat"
#     echo "algo is ${algo}"
#     CUDA_VISIBLE_DEVICES=0 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
#     --n_training_threads 1 --n_rollout_threads 8 --episode_length 400 --num_env_steps 5000000 --num_mini_batch 1 --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval --use_recurrent_policy

#     # algo="mat_dec"
#     # echo "algo is ${algo}"
#     # CUDA_VISIBLE_DEVICES=0 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
#     # --n_training_threads 1 --n_rollout_threads 8 --episode_length 400 --num_env_steps 5000000 --num_mini_batch 1 --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval --use_recurrent_policy

#     # algo="rmappo"
#     # echo "algo is ${algo}"
#     # CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
#     # --n_training_threads 1 --n_rollout_threads 8 --episode_length 400 --num_env_steps 5000000 --num_mini_batch 1 --ppo_epoch 5 --clip_param 0.2 --gain 0.01 --use_value_active_masks --use_eval

# done

#!/bin/sh
env="StarCraft2"
map="27m_vs_30m"
exp="0826"
seed_max=1

echo "env is ${env}, map is ${map}, exp is ${exp}, max seed is ${seed_max}"
for seed in {3..3..1}
do
    echo "seed is ${seed}:"

    # algo="mat"
    # echo "algo is ${algo}"
    # CUDA_VISIBLE_DEVICES=0 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
    # --n_training_threads 1 --n_rollout_threads 8 --episode_length 400 --num_env_steps 10000000 --num_mini_batch 1 --ppo_epoch 5 --clip_param 0.2 --use_value_active_masks --use_eval --use_recurrent_policy

    algo="mat_dec"
    echo "algo is ${algo}"
    CUDA_VISIBLE_DEVICES=0 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 8 --episode_length 400 --num_env_steps 10000000 --num_mini_batch 1 --ppo_epoch 5 --clip_param 0.2 --use_value_active_masks --use_eval --use_recurrent_policy
    
    # algo="rmappo"
    # echo "algo is ${algo}"
    # CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
    # --n_training_threads 1 --n_rollout_threads 8 --episode_length 400 --num_env_steps 5000000 --num_mini_batch 1 --ppo_epoch 5 --clip_param 0.2 --gain 0.01 --use_value_active_masks --use_eval

done
