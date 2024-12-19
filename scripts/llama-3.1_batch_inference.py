import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

def evaluate_preference_batch(rows, constitution, pipe):
    prompts = []
    for row in rows:
        prompt = row['prompt']
        response_1 = row['response_1']
        response_2 = row['response_2']

        input_text = f"""Given the Constitution below, a prompt, and two responses, choose which response better adheres to the guidelines of the Constitution.

Constitution:
{constitution}

Prompt:
{prompt}

Response 1:
"{response_1}"

Response 2:
"{response_2}"

Based on the Constitution, which response better adheres to the guidelines? Respond with "0" if Response 1 is better or "1" if Response 2 is better. Please respond with only a 0 or a 1 and nothing else.
"""
        prompts.append(input_text)

    # Generate the model's responses in batch
    outputs = pipe(prompts, max_new_tokens=10, return_full_text=False)
    print(outputs)

    preferences = []
    for output_list in outputs:
        # Each output_list is a list of generated outputs for a prompt
        generated_text = output_list[0]['generated_text']
        preference = ''.join(filter(str.isdigit, generated_text.strip()))
        if preference in ['0', '1']:
            preferences.append(int(preference))
        else:
            preferences.append(None)

    return preferences

if __name__ == "__main__":
    # Load the data
    const_number = 1
    direct = f"data/preferences/{const_number}"
    all_files = os.listdir(direct)
    df = pd.concat([pd.read_csv(os.path.join(direct, file)) for file in all_files])
    df.info()
    # df = df.head()  # Uncomment for testing with a small dataset

    # Read the Constitution
    with open("data/constitution_1.txt", "r") as f:
        constitution = f.read()

    # Define the model ID and tokenizer
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    # Specify your custom cache directory
    scratch_dir = os.getenv("SCRATCH")  # Use environment variable or default
    cache_directory = os.path.join(scratch_dir, ".cache")

    # Ensure the cache directory exists
    os.makedirs(cache_directory, exist_ok=True)

    # Define the quantization configuration for 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',   # Choices: 'nf4', 'fp4'
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16  # Options: torch.float16 or torch.bfloat16
    )

    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise ValueError("At least two GPUs are required to load the 70B model.")

    # Define max memory per GPU
    gpu_info = [torch.cuda.get_device_properties(i) for i in range(num_gpus)]
    total_memory_per_gpu_in_GB = [int(gpu.total_memory / (1024 ** 3)) for gpu in gpu_info]  # in GB
    max_memory = {i: f"{int(total_memory_per_gpu_in_GB[i]*0.85)}GiB" for i in range(num_gpus)}  # Use 85% of GPU memory

    # Load the tokenizer with the specified cache directory
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_directory
    )

    # Load the model in 4-bit mode across multiple GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,  # Use float16 for computation
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        cache_dir=cache_directory
    )

    # Set up the pipeline with the specified device map
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",  # Automatically maps model layers to available GPUs
        torch_dtype=torch.float16
    )

    # Process the data in batches
    batch_size = 8  # Adjust batch size based on available memory

    preferences = []
    total_batches = (len(df) + batch_size - 1) // batch_size  # Ceiling division
    for idx in range(total_batches):
        start_idx = idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]
        batch_preferences = evaluate_preference_batch(batch.to_dict('records'), constitution, pipe)
        preferences.extend(batch_preferences)
        print(f"Processed batch {idx + 1}/{total_batches}")

    df['preference'] = preferences

    # Save or print the results
    df.to_csv('inference_results.csv', index=False)
    print(df.head())
    print("Inference completed and results saved to 'inference_results.csv'")