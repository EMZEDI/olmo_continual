#!/bin/bash
#SBATCH --job-name=ppo_training_new
#SBATCH --output=log/ppo_training_new_%j.out
#SBATCH --error=log/ppo_training_new_%j.err
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --partition=short-unkillable
#SBATCH --mail-user=   
#SBATCH --mail-type=END,FAIL

source .env

accelerate launch --config_file scripts/configs/PPO_deepspeed.yaml \
    scripts/PPO/ppo.py \
    --dataset_name Shahradmz/ca_constitution_2 \
    --dataset_train_split train \
    --output_dir /network/scratch/s/shahrad.mohammadzadeh/OLMo-1B-hf-PPO-constitution-2 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 15000 \
    --model_name_or_path Shahradmz/OLMo-1B-hf-PPO-constitution-1 \
    --sft_model_path Shahradmz/OLMo-1B-hf-PPO-constitution-1 \
    --reward_model_path Shahradmz/Qwen2.5-0.5B-Reward-LoRA-constitution-2 \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0 \
    --wandb True     --wandb_project OLMo-1B-hf-PPO-constitution-2 \
    --push_to_hub True
