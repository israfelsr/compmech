#!/bin/bash

#SBATCH --job-name=create_dataset  # A descriptive name for your job
#SBATCH --output=create_dataset_%j.out # Standard output file (stdout)
#SBATCH --ntasks=1                   # Request 1 task (process)
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00              # Set a maximum runtime of 10 minutes (HH:MM:SS)

#SBATCH --partition=cpu

export PYTHONPATH=$(pwd)
echo "Current directory: $(pwd)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running Python script..."

python scripts/create_dataset.py
echo "Job finished at: $(date)"