#!/bin/bash
#SBATCH --account EUHPC_D27_102
#SBATCH --job-name=vllm_inference
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1            # 1 GPU
#SBATCH --cpus-per-task=8       # 1/4 of 32 CPUs
#SBATCH --mem=120G              # ~1/4 of total RAM (≈480 GB / 4)
#SBATCH --time=06:00:00         # walltime (hh:mm:ss)
#SBATCH --output=%x_%j.out

# Run your job
export PYTHONPATH=$(pwd)
python scripts/prompting.py \
    --config config/Qwen2.5-VL-3B-Instruct.yaml
