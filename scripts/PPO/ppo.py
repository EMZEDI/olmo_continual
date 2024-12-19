# From TRL Github Repository:
from dataclasses import dataclass
import os
import shutil
from typing import Optional

import torch
from accelerate import PartialState, Accelerator
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    RLOOTrainer
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


"""Usage example

accelerate launch --config_file scripts/configs/PPO_deepspeed.yaml \
    scripts/PPO/ppo.py \
    --dataset_name Shahradmz/ca_constitution_1 \
    --dataset_train_split train \
    --output_dir "$SCRATCH"/OLMo-1B-hf-PPO-constitution-1 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path allenai/OLMo-1B-hf \
    --sft_model_path allenai/OLMo-1B-hf \
    --reward_model_path Shahradmz/Qwen2.5-0.5B-Reward-LoRA-constitution-1 \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0 \
    --wandb True \
    --wandb_project OLMo-1B-hf-PPO-constitution-1 \
    --push_to_hub True
    
"""

@dataclass
class PPOScriptArguments(ScriptArguments):
    cache_dir: str = os.getenv("SCRATCH")
    wandb: bool = True
    wandb_project: str = ""

class FixZero3CheckpointPPOTrainer(PPOTrainer):

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.policy_and_value
        self.policy_and_value = self.policy_and_value  # save only the policy

        Trainer.save_model(self, output_dir, _internal_call)

        self.policy_and_value = backup_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.is_deepspeed_enabled:
            state_dict = {name.removeprefix('policy.'): param for name, param in state_dict.items()
                          if name.startswith('policy.')}

        super()._save(output_dir, state_dict)

if __name__ == "__main__":
    # initialize accelerator
    accelerator = Accelerator()

    parser = HfArgumentParser((PPOScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.report_to = ["wandb"] if script_args.wandb else None
    training_args.run_name = f"{model_config.model_name_or_path.split('/')[-1]}-{script_args.dataset_name}" if script_args.wandb_project else None

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
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        cache_dir=script_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )

    peft_config = get_peft_config(model_config)
    # if "olmo" in model_config.model_name_or_path.lower():
    #     peft_config.target_modules = ["q_proj", "v_proj"]
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_config.trust_remote_code
        )
    # else:
    #     ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, split=script_args.dataset_train_split)
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    trainer = FixZero3CheckpointPPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    try:
        if training_args.push_to_hub:
            # Ensure the repository name is set; if not, derive it from output_dir or provide manually
            repo_name = script_args.wandb_project if script_args.wandb_project else "PPO-Model"
            trainer.push_to_hub(
                repo_id=repo_name,
                use_auth_token=True,  # Assumes you have set up your HuggingFace token
                commit_message="Add PPO fine-tuned model"
            )

    except Exception as e:
        print(f"Failed to push to Hub: {e}")
        pass

    trainer.generate_completions()