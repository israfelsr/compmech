#!/bin/bash
#SBATCH --account EUHPC_D27_102
#SBATCH --job-name=probe_layers
#SBATCH --partition=lrd_all_serial
#SBATCH --cpus-per-task=8       # 1/4 of 32 CPUs
#SBATCH --mem=30G
#SBATCH --time=04:00:00         # walltime (hh:mm:ss)
#SBATCH --output=%x_%j.out


python scripts/train_probes.py --config config/Qwen2.5-VL-7B-Instruct.yaml --layer 20 --prefix vision_