import os
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import multiprocessing as mp

def evaluate_preference_batch(rows, constitution, pipe):
    prompts = []
    for row in rows:
        prompt = row['prompt']
        response_1 = row['response_1']
        response_2 = row['response_2']

        input_text = f""" Given a Constituition which starts in the next line, and a prompt and two responses that will follow, choose which response better adheres to the guidelines of the constitution.
        Constitution:{constitution}

        Prompt and Responses:

        Prompt: {prompt}

        Two responses:
        Response 1: "{response_1}"
        Response 2: "{response_2}"

        Based on the constitution, which response better adheres to the guidelines? Respond with "0" if Response 1 is better or "1" if Response 2 is better. I only want to see a 0 or a 1 nothing else.
        """
        prompts.append(input_text)

    # Generate the model's responses in batch
    outputs = pipe(prompts, max_new_tokens=10, return_full_text=False, num_return_sequences=1)
    print(outputs)

    preferences = []
    for output_list in outputs:
        # output_list is a list of generated outputs for a prompt
        generated_text = output_list[0]['generated_text']
        preference = ''.join(filter(str.isdigit, generated_text.strip()))
        if preference in ['0', '1']:
            preferences.append(int(preference))
        else:
            preferences.append(None)
    
    return preferences

def worker(device_id, df_chunk, constitution, return_dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
    ).to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
    )

    batch_size = 8
    preferences = []
    for i in range(0, len(df_chunk), batch_size):
        batch = df_chunk.iloc[i:i+batch_size]
        batch_preferences = evaluate_preference_batch(batch.to_dict('records'), constitution, pipe)
        preferences.extend(batch_preferences)

    df_chunk = df_chunk.copy()
    df_chunk['preference'] = preferences
    return_dict[device_id] = df_chunk

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    NUM_runs = 200
    NUM_classes = 10
    const_number = 1

    direct = f"data/preferences/{const_number}"
    all_files = os.listdir(direct)
    df = pd.concat([pd.read_csv(f"{direct}/{file}") for file in all_files])
    df.info()
    df = df.head()  # for testing

    load_dotenv("../.env")

    with open("data/constitution_1.txt", "r") as f:
        constitution = f.read()

    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("No GPUs available.")

    df_chunks = np.array_split(df, num_gpus)

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=worker, args=(i, df_chunks[i], constitution, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    df_results = pd.concat(return_dict.values(), ignore_index=True)
    print(df_results)