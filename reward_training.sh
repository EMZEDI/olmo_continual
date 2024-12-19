#!/bin/bash
#SBATCH --job-name=reward_model_training             # Job name
#SBATCH --ntasks=2                                   # Number of tasks
#SBATCH --cpus-per-task=4                            # Number of CPU cores per task
#SBATCH --mem-per-cpu=6G                                    # Memory per task
#SBATCH --gpus-per-task=a100l:1                         # Request 2 GPUs
#SBATCH --partition=main                             # Partition name
#SBATCH --time=12:00:00                              # Time limit hrs:min:sec
#SBATCH --output=log/reward_training_%j.out          # Standard output and error log
#SBATCH --mail-user=                # Email notifications

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK ./run_reward_training.sh "2" &
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK ./run_reward_training.sh "1" &
wait
echo "All training tasks have completed."