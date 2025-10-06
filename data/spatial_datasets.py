import json
import os
from pathlib import Path
from datasets import Dataset, Image as HFImage
import argparse
from tqdm import tqdm


# Dataset configurations
DATASETS = {
    "vg_two_obj": {
        "json": "vg_qa_two_obj.json",
        "images": "vg_images",
        "description": "Visual Genome - Two objects (left/right)",
    },
    "vg_one_obj": {
        "json": "vg_qa_one_obj.json",
        "images": "vg_images",
        "description": "Visual Genome - One object (left/right)",
    },
    "controlled_images": {
        "json": "controlled_images_dataset.json",
        "images": "controlled_images",
        "description": "Controlled images - Two objects (left/right/up/down)",
    },
    "coco_two_obj": {
        "json": "coco_qa_two_obj.json",
        "images": "val2017",
        "description": "COCO - Two objects (up/down/left/right)",
    },
    "coco_one_obj": {
        "json": "coco_qa_one_obj.json",
        "images": "val2017",
        "description": "COCO - One object (up/down/left/right)",
    },
    "controlled_clevr": {
        "json": "controlled_clevr_dataset.json",
        "images": "controlled_clevr",
        "description": "Controlled CLEVR (front/behind/left/right)",
    },
}


def is_dict_format(item):
    """Check if dataset item is in dictionary format"""
    return isinstance(item, dict) and "caption_options" in item


def get_image_id(item):
    """Extract image ID from either format"""
    if is_dict_format(item):
        return item["image_path"]
    else:
        return item[0]


def get_captions(item):
    """Extract all captions from either format (correct first, then incorrect)"""
    if is_dict_format(item):
        return item["caption_options"]
    else:
        # List format: [image_id, correct_caption, incorrect_caption, ...]
        return [item[i] for i in range(1, len(item))]


def get_image_path(base_path, image_id, dataset_key, images_dir):
    """Get full path to image based on dataset conventions"""
    # For dict format, image_id is already a relative path
    if isinstance(image_id, str) and (
        "/" in image_id or image_id.endswith(".jpg") or image_id.endswith(".jpeg")
    ):
        # It's already a path, might need to adjust
        if image_id.startswith("data/"):
            # Remove 'data/' prefix and use base_path
            return str(base_path / image_id.replace("data/", ""))
        return str(base_path / images_dir / Path(image_id).name)

    # For list format with numeric IDs
    if "coco" in dataset_key:
        # COCO uses zero-padded 12-digit IDs
        filename = f"{str(image_id).zfill(12)}.jpg"
    else:
        # Other datasets may use different conventions
        filename = f"{image_id}.jpg" if not str(image_id).endswith(".jpg") else image_id

    return str(base_path / images_dir / filename)


def convert_dataset(base_path, dataset_key, output_dir):
    """
    Convert a single dataset to unified HuggingFace format.

    Args:
        base_path: Path to the base data directory
        dataset_key: Key identifying which dataset to convert
        output_dir: Directory to save the converted dataset

    Returns:
        Dataset object
    """
    print(f"\n{'='*60}")
    print(f"Converting: {DATASETS[dataset_key]['description']}")
    print(f"{'='*60}")

    config = DATASETS[dataset_key]
    json_path = base_path / config["json"]
    images_dir = config["images"]

    # Load JSON data
    print(f"Loading from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Convert to unified format
    unified_data = {"image_path": [], "captions": [], "label": []}

    skipped = 0
    for item in tqdm(data, desc="Processing"):
        # Extract data
        image_id = get_image_id(item)
        captions = get_captions(item)
        img_path = get_image_path(base_path, image_id, dataset_key, images_dir)

        # Check if image exists
        if not Path(img_path).exists():
            skipped += 1
            continue

        # Unified format: correct caption is always first (label = 0)
        unified_data["image_path"].append(img_path)
        unified_data["captions"].append(captions)
        unified_data["label"].append(0)  # First caption is always correct

    if skipped > 0:
        print(f"⚠️  Skipped {skipped} samples due to missing images")

    # Create HuggingFace Dataset
    print("Creating HuggingFace dataset...")
    dataset = Dataset.from_dict(unified_data)

    # Add metadata
    dataset = dataset.add_column("dataset_name", [dataset_key] * len(dataset))
    dataset = dataset.add_column("description", [config["description"]] * len(dataset))

    # Save dataset
    output_path = Path(output_dir) / f"{dataset_key}.hf"
    print(f"Saving to: {output_path}")
    dataset.save_to_disk(str(output_path))

    print(f"✓ Saved {len(dataset)} samples")

    # Print sample
    print("\nSample entry:")
    sample = dataset[0]
    print(f"  Image path: {sample['image_path']}")
    print(f"  Captions ({len(sample['captions'])} options):")
    for i, cap in enumerate(sample["captions"]):
        marker = "✓" if i == sample["label"] else "✗"
        print(f"    {marker} [{i}] {cap}")
    print(f"  Label: {sample['label']}")
    print(f"  Dataset: {sample['dataset_name']}")

    return dataset


def load_unified_dataset(dataset_path):
    """
    Load a unified dataset from disk.

    Args:
        dataset_path: Path to the .hf dataset directory

    Returns:
        Dataset object
    """
    from datasets import load_from_disk

    return load_from_disk(dataset_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert What's Up VLMs datasets to unified HuggingFace format"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data/",
        help="Base path to the whatsup_vlms_data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data/hf/",
        help="Output directory for converted datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
        help="Which datasets to convert (default: all)",
    )

    args = parser.parse_args()

    base_path = Path(args.base_path)
    output_dir = Path(args.output_dir)

    # Validate paths
    if not base_path.exists():
        raise ValueError(f"Base path does not exist: {base_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine which datasets to convert
    if "all" in args.datasets:
        datasets_to_convert = list(DATASETS.keys())
    else:
        datasets_to_convert = args.datasets

    print(f"\nConverting {len(datasets_to_convert)} datasets:")
    for ds in datasets_to_convert:
        print(f"  - {ds}: {DATASETS[ds]['description']}")

    # Convert each dataset
    converted_datasets = {}
    for dataset_key in datasets_to_convert:
        try:
            dataset = convert_dataset(base_path, dataset_key, output_dir)
            converted_datasets[dataset_key] = dataset
        except Exception as e:
            print(f"\n❌ Error converting {dataset_key}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(
        f"Successfully converted: {len(converted_datasets)}/{len(datasets_to_convert)} datasets"
    )

    total_samples = sum(len(ds) for ds in converted_datasets.values())
    print(f"Total samples: {total_samples}")

    print("\nDataset sizes:")
    for name, ds in converted_datasets.items():
        print(f"  {name}: {len(ds)} samples")

    print(f"\nDatasets saved to: {output_dir}")


if __name__ == "__main__":
    main()
