# From TRL Github Repository:
# For Hyperparameter tuning, we can use the following script and https://huggingface.co/docs/transformers/hpo_train 
from dataclasses import dataclass
import warnings
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

"""Usage example
CUDA_VISIBLE_DEVICES=0 python scripts/reward/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --cache_dir "$SCRATCH" \
    --dataset_name Shahradmz/ca_constitution_1 \
    --output_dir "$SCRATCH"/Qwen2.5-0.5B-Reward-LoRA-constitution-1 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-4 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --use_peft \
    --lora_task_type SEQ_CLS \
    --lora_dropout 0.1 \
    --warmup_steps 20 \
    --lora_r 32 \
    --lora_alpha 16 \
    --wandb True \
    --wandb_project Qwen2.5-0.5B-Reward-LoRA-constitution-1 \
    --push_to_hub True
"""

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

import os 

@dataclass
class RewardScriptArguments(ScriptArguments):
    cache_dir: str = os.getenv("SCRATCH")
    wandb: bool = True
    wandb_project: str = ""

if __name__ == "__main__":
    parser = HfArgumentParser((RewardScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    
    # Set the WANDB_PROJECT environment variable if wandb is enabled
    if script_args.wandb and script_args.wandb_project:
        os.environ["WANDB_PROJECT"] = script_args.wandb_project
        print(f"WANDB_PROJECT set to: {script_args.wandb_project}")
    
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.report_to = ["wandb"] if script_args.wandb else None
    training_args.run_name = f"{model_config.model_name_or_path.split('/')[-1]}-{script_args.dataset_name}" if script_args.wandb_project else None
    training_args.center_rewards_coefficient = 0.1

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
        cache_dir=script_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT."
        )

    ##############
    # Load dataset
    ##############
    dataset = load_dataset(script_args.dataset_name) 
    # Keep only 'chosen' and 'rejected' columns
    dataset = dataset.remove_columns(['prompt'])   

    ##########
    # Training
    ##########
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)

    try:
        if training_args.push_to_hub:
            # Ensure the repository name is set; if not, derive it from output_dir or provide manually
            repo_name = script_args.wandb_project if script_args.wandb_project else "Reward-Model"
            trainer.push_to_hub(
                repo_id=repo_name,
                use_auth_token=True,  # Assumes you have set up your HuggingFace token
                commit_message="Add Reward fine-tuned model with LoRA"
            )

    except Exception as e:
        print(f"Failed to push to Hub: {e}")
        pass