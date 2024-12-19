#!/bin/bash
#SBATCH --job-name=dpo_training             # Job name
#SBATCH --output=log/dpo_training_%j.out    # Standard output log
#SBATCH --error=log/dpo_training_%j.err     # Standard error log
#SBATCH --partition=unkillable                    # Partition name
#SBATCH --gres=gpu:a100l:1                  # Request 1 A100L GPU
#SBATCH --mem=32G                           # Total memory
#SBATCH --cpus-per-task=6                   # Number of CPU cores per task
#SBATCH --time=06:00:00                     # Time limit hrs:min:sec
#SBATCH --mail-user=       # Email notifications
#SBATCH --mail-type=END,FAIL                # Notify when job ends or fails

source .env

# Run the DPO script
python scripts/DPO/dpo.py \
    --dataset_name Shahradmz/ca_constitution_2 \
    --model_name_or_path Shahradmz/OLMo-1B-hf-DPO-constitution-1 \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --output_dir "$SCRATCH"/OLMo-1B-hf-DPO-constitution-2 \
    --no_remove_unused_columns \
    --use_peft \
    --lora_task_type SEQ_CLS \
    --lora_dropout 0.1 \
    --warmup_steps 20 \
    --lora_r 32 \
    --lora_alpha 16 \
    --wandb True \
    --wandb_project DPO-OLMo-1B-hf-constitution-2 \
    --push_to_hub True