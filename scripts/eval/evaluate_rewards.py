import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# Configuration
TASKS = [
    {
        "name": "Constitution 1",
        "dataset": "Shahradmz/ca_constitution_1",
        "test_split": "test",
        "models": [
            "Shahradmz/OLMo-1B-hf-DPO-constitution-1",
            "Shahradmz/OLMo-1B-hf-PPO-constitution-1"
        ],
        "reward_model": "Shahradmz/Qwen2.5-0.5B-Reward-LoRA-constitution-1"
    },
    {
        "name": "Constitution 2",
        "dataset": "Shahradmz/ca_constitution_2",
        "test_split": "test",
        "models": [
            "Shahradmz/OLMo-1B-hf-DPO-constitution-2",
            "Shahradmz/OLMo-1B-hf-PPO-constitution-2"
        ],
        "reward_model": "Shahradmz/Qwen2.5-0.5B-Reward-LoRA-constitution-2"
    },
]

OUTPUT_DIR = "rewards_outputs"


def normalize_rewards(rewards):
    min_reward, max_reward = np.min(rewards), np.max(rewards)
    return (rewards - min_reward) / (max_reward - min_reward)


def evaluate_rewards(model_name, dataset_name, test_split, reward_model_name, batch_size=256):
    print(f"Evaluating {model_name} on {dataset_name} using {reward_model_name}")

    # Load reward model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    reward_model = AutoModelForCausalLM.from_pretrained(
        reward_model_name, torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    ).to(device)

    # Load dataset prompts
    dataset = load_dataset(dataset_name, split=test_split)
    prompts = dataset["prompt"]

    # Load generated outputs from CSV
    model_output_dir = os.path.join(
        OUTPUT_DIR,
        model_name.replace('/', '_'),
        dataset_name.replace('/', '_')
    )
    output_file = os.path.join(model_output_dir, 'generated_outputs.csv')
    if not os.path.exists(output_file):
        print(f"Generated outputs not found for {model_name} on {dataset_name}")
        return None
    df = pd.read_csv(output_file)
    generated_outputs = df['generated_text'].tolist()

    # Compute rewards
    rewards = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Calculating Rewards"):
        batch_prompts = prompts[i:i + batch_size]
        batch_generated = generated_outputs[i:i + batch_size]
        inputs = reward_tokenizer(
            [p + o for p, o in zip(batch_prompts, batch_generated)],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = reward_model(**inputs)
            # Process logits to extract scalar reward
            batch_rewards = outputs.logits.mean(dim=-1).mean(dim=-1).cpu().numpy()
            rewards.extend(batch_rewards)

        # Clear memory
        del inputs, outputs
        torch.cuda.empty_cache()

    # Normalize and return rewards
    normalized_rewards = normalize_rewards(np.array(rewards))
    avg_reward = np.mean(normalized_rewards)
    print(f"Average normalized reward: {avg_reward:.4f}")
    return avg_reward


# Main execution
if __name__ == "__main__":
    results = {}
    for i, task in enumerate(TASKS):
        task_name = task["name"]
        results[task_name] = {}
        for model_name in task["models"]:
            model_results = {}
            # Evaluate on current dataset
            current_avg_reward = evaluate_rewards(
                model_name, task["dataset"], task["test_split"], task["reward_model"]
            )
            model_results[task_name] = current_avg_reward
            # Evaluate on previous datasets if any
            if i > 0:
                for previous_task in TASKS[:i]:
                    prev_task_name = previous_task["name"]
                    prev_avg_reward = evaluate_rewards(
                        model_name, previous_task["dataset"], previous_task["test_split"], previous_task["reward_model"]
                    )
                    model_results[prev_task_name] = prev_avg_reward
            results[task_name][model_name] = model_results

    # Print summary
    print("\n--- Summary of Results ---")
    for task_name, task_results in results.items():
        print(f"\nTask: {task_name}")
        for model_name, model_results in task_results.items():
            print(f"  Model: {model_name}")
            for eval_task, avg_reward in model_results.items():
                print(f"    {eval_task}: {avg_reward:.4f}")
