#!/bin/bash
#SBATCH --job-name=data_generation     # Job name
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=6              # Number of CPU cores per task
#SBATCH --mem=32G                      # Total memory for the job
#SBATCH --partition=main               # Partition name
#SBATCH --time=10:00:00                # Time limit hrs:min:sec

source .env

# Loop over contexts from 1 to 10
for context in {1..10}; do
    # Run the data_generation.py script with the current context and constitution 1
    python scripts/data_generation.py --context $context --constitution 2
done