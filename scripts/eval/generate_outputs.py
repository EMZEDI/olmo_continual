import os
import torch
import pandas as pd
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
    },
    {
        "name": "Constitution 2",
        "dataset": "Shahradmz/ca_constitution_2",
        "test_split": "test",
        "models": [
            "Shahradmz/OLMo-1B-hf-DPO-constitution-2",
            "Shahradmz/OLMo-1B-hf-PPO-constitution-2"
        ],
    },
]

# Directory to save outputs
OUTPUT_DIR = "rewards_outputs"

def generate_outputs(model, tokenizer, prompts, output_file, device, batch_size=256):
    outputs = []
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Outputs", leave=False):
        batch_prompts = prompts[i:i+batch_size]
        encoding = tokenizer(batch_prompts, padding=True, return_tensors='pt', truncation=True).to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding,
                max_length=512,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )
        batch_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        outputs.extend(batch_outputs)
    
    # Save outputs to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame({'prompt': prompts, 'generated_text': outputs})
    df.to_csv(output_file, index=False, encoding='utf-8')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for task in tqdm(TASKS, desc="Tasks"):
        dataset = load_dataset(task["dataset"], split=task["test_split"])
        prompts = dataset["prompt"]
        for model_name in tqdm(task["models"], desc=f"Models for {task['name']}", leave=False):
            print(f"\nProcessing model {model_name}")
            # Load model and tokenizer once per model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32).to(device)
            
            # Generate outputs for current task
            output_file = os.path.join(
                OUTPUT_DIR,
                model_name.replace('/', '_'),
                task["dataset"].replace('/', '_'),
                'generated_outputs.csv'
            )
            print(f"Generating outputs for {model_name} on {task['dataset']}")
            generate_outputs(model, tokenizer, prompts, output_file, device)
            
            # Generate outputs for previous tasks if any
            for previous_task in TASKS:
                if previous_task == task:
                    break
                prev_dataset = load_dataset(previous_task["dataset"], split=previous_task["test_split"])
                prev_prompts = prev_dataset["prompt"]
                prev_output_file = os.path.join(
                    OUTPUT_DIR,
                    model_name.replace('/', '_'),
                    previous_task["dataset"].replace('/', '_'),
                    'generated_outputs.csv'
                )
                print(f"Generating outputs for {model_name} on {previous_task['dataset']}")
                generate_outputs(model, tokenizer, prev_prompts, prev_output_file, device)