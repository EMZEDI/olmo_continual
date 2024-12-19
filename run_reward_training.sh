#!/bin/bash
TASK_ID=$1

# Define base arguments
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B"
CACHE_DIR="$SCRATCH"
DATASET_NAME="Shahradmz/ca_constitution_"
PER_DEVICE_BATCH_SIZE=8
NUM_TRAIN_EPOCHS=1
GRADIENT_CHECKPOINTING=True
LEARNING_RATE=1.0e-4
LOGGING_STEPS=10
EVAL_STRATEGY="steps"
EVAL_STEPS=25
MAX_LENGTH=2048
USE_PEFT=True
LORA_TASK_TYPE="SEQ_CLS"
LORA_DROPOUT=0.1
WARMUP_STEPS=20
LORA_R=32
LORA_ALPHA=16
WANDB=True

# Define unique output directory and wandb project for each task
OUTPUT_DIR="$SCRATCH/Qwen2.5-0.5B-Reward-LoRA-constitution-${TASK_ID}"
WANDB_PROJECT="Qwen2.5-0.5B-Reward-LoRA-constitution-${TASK_ID}"

echo "Running reward modeling training for task ${TASK_ID}..."

source .env

# Run the training script
python scripts/reward/reward_modeling.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --cache_dir $CACHE_DIR \
    --dataset_name "${DATASET_NAME}${TASK_ID}" \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --learning_rate $LEARNING_RATE \
    --logging_steps $LOGGING_STEPS \
    --eval_strategy $EVAL_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --max_length $MAX_LENGTH \
    --use_peft $USE_PEFT \
    --lora_task_type $LORA_TASK_TYPE \
    --lora_dropout $LORA_DROPOUT \
    --warmup_steps $WARMUP_STEPS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --wandb $WANDB \
    --wandb_project $WANDB_PROJECT