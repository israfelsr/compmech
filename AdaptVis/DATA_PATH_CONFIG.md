# Data Path Configuration

This document explains how to configure the data paths for the AdaptVis evaluation framework.

## Changes Made

The code has been updated to support custom data directories for all datasets. Previously, it defaulted to looking for data in the `data/` directory.

### Modified Files

1. **`dataset_zoo/aro_datasets.py`**:
   - Updated all `get_*` helper functions to accept `root_dir` parameter
   - Modified `Controlled_Images.__getitem__()` to handle both absolute and relative paths

2. **`main_aro.py`**:
   - Added `--data-dir` argument (default: `/leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data`)
   - Pass `root_dir` to `get_dataset()`

3. **`run_qwen_vllm_eval.sh`**:
   - Added `DATA_DIR` variable
   - Pass `--data-dir` to python command

## Data Directory Structure

Your data should be organized as follows:

```
/leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data/
├── coco_qa_one_obj.json          # COCO one-object annotations
├── coco_qa_two_obj.json          # COCO two-object annotations
├── vg_qa_one_obj.json            # VG one-object annotations
├── vg_qa_two_obj.json            # VG two-object annotations
├── controlled_images_dataset.json # Controlled images annotations
├── controlled_clevr_dataset.json  # Controlled CLEVR annotations
├── val2017/                       # COCO images
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
├── vg_images/                     # Visual Genome images
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── controlled_images/             # Controlled images
│   ├── apple_left_of_banana_0.jpeg
│   └── ...
└── controlled_clevr/              # Controlled CLEVR images
    ├── cube_left_of_sphere_0.png
    └── ...
```

## Usage

### Command Line

You can now specify a custom data directory:

```bash
python AdaptVis/main_aro.py \
    --model-name qwen2vl-vllm \
    --dataset Controlled_Images_A \
    --option four \
    --data-dir /path/to/your/data \
    --device cuda
```

### Default Behavior

If you don't specify `--data-dir`, it defaults to:
```
/leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data
```

### Using the Script

The evaluation script automatically uses the correct data directory:

```bash
./run_qwen_vllm_eval.sh
```

To change the data directory in the script, edit the `DATA_DIR` variable:

```bash
DATA_DIR="/your/custom/path"
```

## Dataset-Specific Paths

### COCO Datasets
- JSON files: `coco_qa_one_obj.json`, `coco_qa_two_obj.json`
- Images: `{data_dir}/val2017/{image_id}.jpg`

### Visual Genome Datasets
- JSON files: `vg_qa_one_obj.json`, `vg_qa_two_obj.json`
- Images: `{data_dir}/vg_images/{image_id}.jpg`

### Controlled Images
- JSON file: `controlled_images_dataset.json`
- Images: Paths specified in JSON (can be relative or absolute)
  - Relative paths are joined with `{data_dir}/`
  - Absolute paths are used as-is

### Controlled CLEVR
- JSON file: `controlled_clevr_dataset.json`
- Images: Paths specified in JSON (can be relative or absolute)
  - Relative paths are joined with `{data_dir}/`
  - Absolute paths are used as-is

## Troubleshooting

### Image Not Found Errors

If you get "Image not found" or "No such file or directory" errors:

1. **Check data directory exists**:
   ```bash
   ls -la /leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data/
   ```

2. **Check image subdirectories exist**:
   ```bash
   ls -la /leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data/val2017/
   ls -la /leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data/vg_images/
   ls -la /leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data/controlled_images/
   ```

3. **Verify JSON files exist**:
   ```bash
   ls -la /leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data/*.json
   ```

4. **Check JSON contains correct paths**:
   ```bash
   # For controlled images
   head -1 /leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data/controlled_images_dataset.json | jq '.image_path'
   ```

### Permission Errors

If you get permission errors:
```bash
chmod -R 755 /leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data/
```

### JSON Path Issues

If the JSON files have paths like `data/controlled_images/...`, they will be automatically converted to:
`/leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data/controlled_images/...`

The code handles both relative and absolute paths intelligently.

## Examples

### Run on Controlled Images A
```bash
python AdaptVis/main_aro.py \
    --model-name qwen2vl-vllm \
    --dataset Controlled_Images_A \
    --option four \
    --data-dir /leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data
```

### Run on COCO One Object
```bash
python AdaptVis/main_aro.py \
    --model-name qwen2vl-vllm \
    --dataset COCO_QA_one_obj \
    --option four \
    --data-dir /leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data
```

### Run on VG Two Objects
```bash
python AdaptVis/main_aro.py \
    --model-name qwen2vl-vllm \
    --dataset VG_QA_two_obj \
    --option four \
    --data-dir /leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data
```
