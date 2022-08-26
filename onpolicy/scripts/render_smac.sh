# #!/bin/sh
# env="StarCraft2"
# map="MMM2"
# exp="render"
# seed_max=1
# seed=1
# echo "env is ${env}, map is ${map}, exp is ${exp}, max seed is ${seed_max}"
# echo "seed is ${seed}:"

# algo="rmappo"
# echo "algo is ${algo}"
# CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
# --n_training_threads 1 --n_rollout_threads 32 --episode_length 100 --num_env_steps 10000000 \
# --num_mini_batch 2 --ppo_epoch 5 --gain 1 --use_value_active_masks --use_eval \
# --render --use_render --eval_episodes 1 --use_wandb \
# --model_dir /home/eva_data/xyq/onpolicy/onpolicy/scripts/results/StarCraft2/MMM2/distill/sanity-check/wandb/run-20220819_063943-28b8d6fa/files  --layer_N 1 #/transformer_3124.pt --use_recurrent_policy

#!/bin/sh
env="StarCraft2"
map="27m_vs_30m"
exp="render"
seed_max=1
seed=1
echo "env is ${env}, map is ${map}, exp is ${exp}, max seed is ${seed_max}"
echo "seed is ${seed}:"

algo="mat"
echo "algo is ${algo}"
CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
--n_training_threads 1 --n_rollout_threads 32 --episode_length 100 --num_env_steps 10000000 \
--num_mini_batch 2 --ppo_epoch 5 --gain 1 --use_value_active_masks --use_eval \
--render --use_render --eval_episodes 1 --use_wandb  --layer_N 1 \
--model_dir /home/eva_data/xyq/onpolicy/onpolicy/scripts/results/StarCraft2/27m_vs_30m/mat/0811/wandb/run-20220811_104254-1tid0m1r/files/transformer_3124.pt --use_recurrent_policy
# /home/eva_data/xyq/onpolicy/onpolicy/scripts/results/StarCraft2/27m_vs_30m/rmappo/0811/wandb/run-20220811_101616-324qriz5/files
# /home/eva_data/xyq/onpolicy/onpolicy/scripts/results/StarCraft2/27m_vs_30m/mat/0811/wandb/run-20220811_104254-1tid0m1r/files/transformer_3124.pt
# /home/eva_data/xyq/onpolicy/onpolicy/scripts/results/StarCraft2/27m_vs_30m/distill/mat/wandb/run-20220822_135220-k2w8z18s/files
# --model_dir /home/eva_data/xyq/onpolicy/onpolicy/scripts/results/StarCraft2/MMM2/distill/sanity-check/wandb/run-20220819_063943-28b8d6fa/files  --layer_N 1 #/transformer_3124.pt --use_recurrent_policy
