#!/bin/bash
#SBATCH --job-name=ppo_training_scheduler
#SBATCH --output=log/ppo_trainingscheduler_%j.out
#SBATCH --error=log/ppo_trainingscheduler_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=unkillable-cpu
#SBATCH --mail-user=   
#SBATCH --mail-type=END,FAIL

# First set of names
DATASET_NAME="Shahradmz/ca_constitution_1"
OUTPUT_DIR="$SCRATCH/OLMo-1B-hf-PPO-constitution-1"
WANDB_PROJECT="OLMo-1B-hf-PPO-constitution-1"
MODEL_NAME_OR_PATH="allenai/OLMo-1B-hf"
SFT_MODEL_PATH="allenai/OLMo-1B-hf"
REWARD_MODEL_PATH="Shahradmz/Qwen2.5-0.5B-Reward-LoRA-constitution-1"

# Create the first Slurm job script
cat <<EOT > ppo_job.sh
#!/bin/bash
#SBATCH --job-name=ppo_training
#SBATCH --output=log/ppo_training_%j.out
#SBATCH --error=log/ppo_training_%j.err
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --partition=short-unkillable
#SBATCH --mail-user=   
#SBATCH --mail-type=END,FAIL

source .env

accelerate launch --config_file scripts/configs/PPO_deepspeed.yaml \\
    scripts/PPO/ppo.py \\
    --dataset_name ${DATASET_NAME} \\
    --dataset_train_split train \\
    --output_dir ${OUTPUT_DIR} \\
    --num_ppo_epochs 1 \\
    --num_mini_batches 1 \\
    --learning_rate 1e-6 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 16 \\
    --total_episodes 15000 \\
    --model_name_or_path ${MODEL_NAME_OR_PATH} \\
    --sft_model_path ${SFT_MODEL_PATH} \\
    --reward_model_path ${REWARD_MODEL_PATH} \\
    --local_rollout_forward_batch_size 1 \\
    --missing_eos_penalty 1.0 \\
    --wandb True \\
    --wandb_project ${WANDB_PROJECT} \\
    --push_to_hub True
EOT

# Submit the first job
sbatch ppo_job.sh

# Wait for 3 hours
sleep 3h

# Update names for the second job
DATASET_NAME="Shahradmz/ca_constitution_2"
OUTPUT_DIR="$SCRATCH/OLMo-1B-hf-PPO-constitution-2"
WANDB_PROJECT="OLMo-1B-hf-PPO-constitution-2"
MODEL_NAME_OR_PATH="Shahradmz/OLMo-1B-hf-PPO-constitution-1"
SFT_MODEL_PATH="Shahradmz/OLMo-1B-hf-PPO-constitution-1"
REWARD_MODEL_PATH="Shahradmz/Qwen2.5-0.5B-Reward-LoRA-constitution-2"

# Create the second Slurm job script
cat <<EOT > ppo_job_new.sh
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

accelerate launch --config_file scripts/configs/PPO_deepspeed.yaml \\
    scripts/PPO/ppo.py \\
    --dataset_name ${DATASET_NAME} \\
    --dataset_train_split train \\
    --output_dir ${OUTPUT_DIR} \\
    --num_ppo_epochs 1 \\
    --num_mini_batches 1 \\
    --learning_rate 1e-6 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 16 \\
    --total_episodes 15000 \\
    --model_name_or_path ${MODEL_NAME_OR_PATH} \\
    --sft_model_path ${SFT_MODEL_PATH} \\
    --reward_model_path ${REWARD_MODEL_PATH} \\
    --local_rollout_forward_batch_size 1 \\
    --missing_eos_penalty 1.0 \\
    --wandb True \
    --wandb_project ${WANDB_PROJECT} \\
    --push_to_hub True
EOT

# Submit the second job
sbatch ppo_job_new.sh