# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Usage:

CUDA_VISIBLE_DEVICES=0 python scripts/Online_DPO/online_DPO.py \
    --model_name_or_path allenai/OLMo-1B-hf  \
    --reward_model_path <TBD> \
    --dataset_name Shahradmz/ca_constitution_1 \
    --learning_rate 5.0e-6 \
    --output_dir "$SCRATCH"/OLMo-1B-hf-online-DPO-constitution-1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --use_peft \
    --lora_task_type SEQ_CLS \
    --lora_dropout 0.1 \
    --warmup_steps 20 \
    --lora_r 32 \
    --lora_alpha 16 \
    --wandb True \
    --wandb_project Online-DPO-OLMo-1B-hf-constitution-1 \
    --push_to_hub True
"""

from dataclasses import dataclass
import warnings
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
)
from trl import (
    HfPairwiseJudge,
    LogCompletionsCallback,
    ModelConfig,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    OpenAIPairwiseJudge,
    PairRMJudge,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

import os

@dataclass
class OnlineDPOScriptArguments(ScriptArguments):
    cache_dir: str = os.getenv("SCRATCH")
    wandb: bool = True
    wandb_project: str = ""

JUDGES = {"pair_rm": PairRMJudge, "openai": OpenAIPairwiseJudge, "hf": HfPairwiseJudge}

if __name__ == "__main__":
    parser = TrlParser((OnlineDPOScriptArguments, OnlineDPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    # Set the WANDB_PROJECT environment variable if wandb is enabled
    if script_args.wandb and script_args.wandb_project:
        os.environ["WANDB_PROJECT"] = script_args.wandb_project
        print(f"WANDB_PROJECT set to: {script_args.wandb_project}")

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    training_args.report_to = ["wandb"] if script_args.wandb else None
    if script_args.wandb_project:
        training_args.run_name = f"{model_config.model_name_or_path.split('/')[-1]}-{script_args.dataset_name}"
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
        cache_dir=script_args.cache_dir
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs
    )

    if training_args.reward_model_path is not None:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path,
            num_labels=1,
            trust_remote_code=model_config.trust_remote_code,
            **model_kwargs,
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(
            training_args.reward_model_path,
            trust_remote_code=model_config.trust_remote_code,
            truncation=True,
            truncation_side="left",  # since we judge the completion, truncating left is more appropriate
        )
    else:
        reward_model = None
        reward_tokenizer = None

    if training_args.judge is not None:
        judge_cls = JUDGES.get(training_args.judge.lower())
        if judge_cls is not None:
            judge = judge_cls()
        else:
            raise ValueError(f"Judge '{training_args.judge}' is not recognized. Choose from {list(JUDGES.keys())}.")
    else:
        judge = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if isinstance(tokenizer, AutoTokenizer) and not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(script_args.dataset_name)
    
    # Optional: Remove unwanted columns
    if "prompt" in dataset.column_names["train"]:
        dataset = dataset.remove_columns(["prompt"])

    trainer = OnlineDPOTrainer(
        model=model,
        reward_model=reward_model,
        judge=judge,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        reward_processing_class=reward_tokenizer,
        peft_config=get_peft_config(model_config),
        lora_r=script_args.lora_r if script_args.use_peft else None,
        lora_alpha=script_args.lora_alpha if script_args.use_peft else None,
        lora_dropout=script_args.lora_dropout if script_args.use_peft else None,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)

    try:
        if training_args.push_to_hub:
            # Ensure the repository name is set; if not, derive it from output_dir or provide manually
            repo_name = script_args.wandb_project if script_args.wandb_project else "Online-DPO-Model"
            trainer.push_to_hub(
                repo_id=repo_name,
                use_auth_token=True,  # Assumes you have set up your HuggingFace token
                commit_message="Add Online fine-tuned model with LoRA"
            )

    except Exception as e:
        print(f"Failed to push to Hub: {e}")
        pass

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)