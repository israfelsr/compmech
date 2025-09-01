#!/bin/bash
#SBATCH --account EUHPC_D27_102
#SBATCH --job-name=probe_layers
#SBATCH --partition=lrd_all_serial
#SBATCH --cpus-per-task=8       # 1/4 of 32 CPUs
#SBATCH --mem=120G              # ~1/4 of total RAM (≈480 GB / 4)
#SBATCH --time=04:00:00         # walltime (hh:mm:ss)
#SBATCH --output=%x_%j.out
#SBATCH --array=0-27%2

# Create logs directory if it doesn't exist
mkdir -p logs

# Define the layers to process (0-12 + last)
#layers=("last" 0 1 2 3 4 5 6 7 8 9 10 11 12)
n=27
layers=($(seq 0 $n))

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

python scripts/train_probes.py \
    --config config/paligemma2-3b-ft-docci-448.yaml \
    --layer $current_layer \
    --output_dir results/probes/paligemma2-3b-ft-docci-448/layer_${current_layer} \

echo "Job finished at: $(date)"
echo "Layer $current_layer completed successfully"