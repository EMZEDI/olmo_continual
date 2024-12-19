from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import argparse
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--repo_name", type=str, required=True, help="Hugging Face Hub repository name")
    parser.add_argument("--wandb_run_id", type=str, required=True, help="W&B run ID to restore")
    parser.add_argument("--wandb_project", type=str, required=True, help="W&B project name")
    args = parser.parse_args()

    # Base model name (same as used during training)
    base_model_name = "Qwen/Qwen2.5-0.5B"  # Replace with your base model if different

    # Load the tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Load the base model with the correct number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        trust_remote_code=True,
    )

    # Load the LoRA adapters from the output directory
    model = PeftModel.from_pretrained(
        model,
        args.output_dir,
    )

    # Restore the W&B run
    wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="must")

    # Push the model and tokenizer to the Hugging Face Hub
    model.push_to_hub(args.repo_name)
    tokenizer.push_to_hub(args.repo_name)