#!/bin/bash
#SBATCH --job-name=ppo_training             # Job name
#SBATCH --output=log/ppo_training_%j.out    # Standard output log
#SBATCH --error=log/ppo_training_%j.err     # Standard error log
#SBATCH --gres=gpu:a100l:4                 # Request 4 A100L GPU
#SBATCH --mem=128G                           # Total memory
#SBATCH --cpus-per-task=24            # Number of CPU cores per task
#SBATCH --partition=short-unkillable                    # Partition name
#SBATCH --mail-user=       # Email notifications
#SBATCH --mail-type=END,FAIL                # Notify when job ends or fails

source .env

accelerate launch --config_file scripts/configs/PPO_deepspeed.yaml \
    scripts/PPO/ppo.py \
    --dataset_name Shahradmz/ca_constitution_1 \
    --dataset_train_split train \
    --output_dir "$SCRATCH"/OLMo-1B-hf-PPO-constitution-1 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10 \
    --model_name_or_path allenai/OLMo-1B-hf \
    --sft_model_path allenai/OLMo-1B-hf \
    --reward_model_path Shahradmz/Qwen2.5-0.5B-Reward-LoRA-constitution-1 \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0 \
    --wandb_project OLMo-1B-hf-PPO-constitution-1 \
    --push_to_hub True