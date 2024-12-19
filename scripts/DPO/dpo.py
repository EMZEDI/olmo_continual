# From TRL Github Repository:
from dataclasses import dataclass
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

import os

"""Example usage:
CUDA_VISIBLE_DEVICES=0 python scripts/DPO/dpo.py \
    --dataset_name Shahradmz/ca_constitution_1 \
    --model_name_or_path allenai/OLMo-1B-hf \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --output_dir "$SCRATCH"/OLMo-1B-hf-DPO-constitution-1 \
    --no_remove_unused_columns \
    --use_peft \
    --lora_task_type SEQ_CLS \
    --lora_dropout 0.1 \
    --warmup_steps 20 \
    --lora_r 32 \
    --lora_alpha 16 \
    --wandb True \
    --wandb_project DPO-OLMo-1B-hf-constitution-1 \
    --push_to_hub True
"""

@dataclass
class DPOScriptArguments(ScriptArguments):
    cache_dir: str = os.getenv("SCRATCH", "./cache")
    wandb: bool = True
    wandb_project: str = ""

if __name__ == "__main__":
    parser = HfArgumentParser((DPOScriptArguments, DPOConfig, ModelConfig))
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
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        cache_dir=script_args.cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    if "olmo" in model_config.model_name_or_path.lower():
        peft_config.target_modules = ["q_proj", "v_proj"]
    if peft_config is None:
        print("Applying PEFT (LoRA) configuration.")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # Torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name)
    
    ##########
    # Training
    ##########
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=peft_config,
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
            repo_name = script_args.wandb_project if script_args.wandb_project else "DPO-Model"
            trainer.push_to_hub(
                repo_id=repo_name,
                use_auth_token=True,  # Assumes you have set up your HuggingFace token
                commit_message="Add DPO fine-tuned model with LoRA"
            )
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        pass