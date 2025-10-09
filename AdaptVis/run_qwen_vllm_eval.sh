#!/bin/bash
# Example script to run Qwen2-VL with vLLM on all datasets

# Configuration
MODEL="qwen2vl-vllm"
DEVICE="cuda"
OPTION="four"
OUTPUT_BASE="./results/qwen_vllm"

# Create output directory
mkdir -p $OUTPUT_BASE

# Datasets to evaluate
DATASETS=(
    "Controlled_Images_A"
    "Controlled_Images_B"
    "COCO_QA_one_obj"
    "COCO_QA_two_obj"
    "VG_QA_one_obj"
    "VG_QA_two_obj"
)

echo "Starting Qwen2-VL vLLM evaluation on all datasets..."
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Options: $OPTION"
echo "=============================================="

# Run evaluation on each dataset
for dataset in "${DATASETS[@]}"
do
    echo ""
    echo "Evaluating on dataset: $dataset"
    echo "----------------------------------------------"

    python main_aro.py \
        --model-name $MODEL \
        --dataset $dataset \
        --option $OPTION \
        --device $DEVICE \
        --output-dir "$OUTPUT_BASE/$dataset" \
        --seed 42

    echo "Completed: $dataset"
    echo "=============================================="
done

echo ""
echo "All evaluations completed!"
echo "Results saved to: $OUTPUT_BASE"
