#!/bin/bash
#SBATCH --job-name=probe_all_layers
#SBATCH --output=logs/probe_layers_%A_%a.out
#SBATCH --error=logs/probe_layers_%A_%a.err
#SBATCH --array=0-13
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --partition=cpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Define the layers to process (0-12 + last)
layers=("last" 0 1 2 3 4 5 6 7 8 9 10 11 12)

# Get the current layer from the array index
current_layer=${layers[$SLURM_ARRAY_TASK_ID]}

echo "Processing layer: $current_layer"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Job started at: $(date)"

# Set number of threads for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load required modules (adjust as needed for your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate your environment
# source /home/bzq999/miniconda3/envs/compmech/bin/activate

# Run the probe training for the specific layer
cd /home/bzq999/data/compmech

python scripts/train_probes.py \
    --config config/dinov2.yaml \
    --layer $current_layer \
    --output-dir results/probes/layer_${current_layer} \

echo "Job finished at: $(date)"
echo "Layer $current_layer completed successfully"