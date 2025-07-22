import json
import os
from pathlib import Path
import numpy as np
from datasets import Dataset, DatasetDict
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def scan_images_in_folders(image_dir: str) -> Dict[str, List[str]]:
    """
    Scan the image directory for object folders and collect all image paths.

    Args:
        image_dir: Path to directory containing object folders

    Returns:
        Dictionary mapping object_name -> list of image paths
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    object_to_images = {}

    # Scan for object folders
    for object_folder in image_dir.iterdir():
        if object_folder.is_dir():
            object_name = object_folder.name
            image_paths = []

            # Collect all images in this object folder
            for image_file in object_folder.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tiff",
                ]:
                    image_paths.append(str(image_file))

            if image_paths:
                object_to_images[object_name] = image_paths
            else:
                logging.warning(f"No images found for object '{object_name}'")

    logging.info(f"Scanned {len(object_to_images)} object folders with images")
    return object_to_images


def create_attribute_mapping(
    concept_data: List[List[str]], attribute_data: Dict[str, str]
) -> Tuple[Dict, List[str], int]:
    """
    Create mappings between concepts, attributes, and indices.

    Args:
        concept_data: List of [concept, attribute] pairs
        attribute_data: Dictionary mapping attributes to taxonomies

    Returns:
        Tuple of (concept_to_attributes, all_attributes, num_attributes)
    """
    # Create vocabulary and mappings for attributes
    all_attributes = sorted(list(attribute_data.keys()))
    attribute_to_idx = {attr: i for i, attr in enumerate(all_attributes)}
    num_attributes = len(all_attributes)

    # Process concepts and group attributes by concept
    concept_to_attributes = {}
    for concept, attribute in concept_data:
        if concept not in concept_to_attributes:
            concept_to_attributes[concept] = []
        concept_to_attributes[concept].append(attribute)

    return concept_to_attributes, all_attributes, num_attributes


def create_dataset_samples(
    object_to_images: Dict[str, List[str]],
    concept_to_attributes: Dict[str, List[str]],
    all_attributes: List[str],
) -> Dict[str, List]:
    """
    Create the dataset samples with image paths, object names, and individual binary attributes.

    Args:
        object_to_images: Dictionary mapping object names to image paths
        concept_to_attributes: Dictionary mapping concepts to list of their attributes
        all_attributes: List of all attribute names

    Returns:
        Dictionary with 'image_path', 'concept', and individual attribute columns (att_0, att_1, ...)
    """
    # Initialize sample dictionary with image_path and concept
    samples = {
        "image_path": [],
        "concept": [],
    }

    # Add columns for each attribute (att_0, att_1, ..., att_n)
    for i, attr_name in enumerate(all_attributes):
        samples[attr_name] = []

    # Create attribute to index mapping
    attribute_to_idx = {attr: i for i, attr in enumerate(all_attributes)}

    # Create samples for each image
    for object_name, image_paths in object_to_images.items():
        if object_name in concept_to_attributes:
            # Get list of attributes for this concept
            concept_attributes = concept_to_attributes[object_name]

            # Create binary vector for this concept
            binary_attributes = [0] * len(all_attributes)
            for attr in concept_attributes:
                if attr in attribute_to_idx:
                    idx = attribute_to_idx[attr]
                    binary_attributes[idx] = 1

            # Add one sample per image of this concept
            for image_path in image_paths:
                samples["image_path"].append(image_path)
                samples["concept"].append(object_name)

                # Add binary value for each attribute
                for att, attr_value in zip(all_attributes, binary_attributes):
                    samples[att].append(attr_value)
        else:
            logging.warning(
                f"No attributes found for object '{object_name}', skipping {len(image_paths)} images"
            )

    logging.info(f"Created {len(samples['image_path'])} dataset samples")
    logging.info(
        f"Dataset has {len(all_attributes)} attributes (att_0 to att_{len(all_attributes)-1})"
    )

    return samples


def create_dataset(
    concept_file="dataset/mcrae-x-things.json",
    attribute_file="dataset/mcrae-x-things-taxonomy.json",
    image_dir="/home/bzq999/data/compmech/image_database_things/object_images",
    output_dir="/home/bzq999/data/compmech/mcrae-x-things.hf",
):
    """
    Create a HuggingFace dataset from the concept-attribute data and images.

    Args:
        concept_file: Path to JSON file with concept-attribute pairs
        attribute_file: Path to JSON file with attribute taxonomy
        image_dir: Path to directory containing object folders with images
        output_dir: Directory to save the HuggingFace dataset
        dataset_name: Name for the dataset
        train_split: Proportion for training set
        val_split: Proportion for validation set
        test_split: Proportion for test set

    Returns:
        DatasetDict with train/val/test splits
    """

    logging.info(f"Creating HuggingFace dataset from:")
    logging.info(f"  Concepts: {concept_file}")
    logging.info(f"  Attributes: {attribute_file}")
    logging.info(f"  Images: {image_dir}")

    # Load concept and attribute data
    with open(concept_file, "r") as f:
        concept_data = json.load(f)
    with open(attribute_file, "r") as f:
        attribute_data = json.load(f)

    logging.info(f"Loaded {len(concept_data)} concept-attribute pairs")
    logging.info(f"Loaded {len(attribute_data)} attribute definitions")

    # Scan for images
    object_to_images = scan_images_in_folders(image_dir)

    # Create attribute mappings
    concept_to_attributes, all_attributes, num_attributes = create_attribute_mapping(
        concept_data, attribute_data
    )

    logging.info(f"Processing {num_attributes} attributes")

    # Create dataset samples
    dataset_samples = create_dataset_samples(
        object_to_images, concept_to_attributes, all_attributes
    )

    # Create HuggingFace dataset
    dataset = Dataset.from_dict(dataset_samples)

    logging.info(f"Created dataset with {len(dataset)} samples")
    logging.info(f"Dataset features: {dataset.features}")

    # Save the dataset
    output_path = Path(output_dir)
    dataset.save_to_disk(str(output_path))

    logging.info(f"Dataset saved to: {output_path}")

    # Save metadata
    metadata = {
        "dataset_name": output_path.stem,
        "num_samples": len(dataset),
        "num_attributes": num_attributes,
        "attribute_names": all_attributes,
        "num_objects": len(concept_to_attributes),
        "concept_file": concept_file,
        "attribute_file": attribute_file,
        "image_dir": image_dir,
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Metadata saved to: {metadata_path}")

    return dataset_dict


if __name__ == "__main__":
    # Create the dataset
    dataset_dict = create_dataset(
        concept_file="dataset/mcrae-x-things.json",
        attribute_file="dataset/mcrae-x-things-taxonomy.json",
        image_dir="/home/bzq999/data/compmech/image_database_things/object_images",
    )
