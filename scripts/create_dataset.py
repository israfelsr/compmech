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


def create_concept_only_samples(
    concept_to_attributes: Dict[str, List[str]],
    all_attributes: List[str],
) -> Dict[str, List]:
    """
    Create concept-only dataset samples with only concepts and their binary attributes (no images).

    Args:
        concept_to_attributes: Dictionary mapping concepts to list of their attributes
        all_attributes: List of all attribute names

    Returns:
        Dictionary with 'concept' and individual attribute columns
    """
    # Initialize sample dictionary with concept only
    samples = {
        "concept": [],
    }

    # Add columns for each attribute
    for attr_name in all_attributes:
        samples[attr_name] = []

    # Create attribute to index mapping
    attribute_to_idx = {attr: i for i, attr in enumerate(all_attributes)}

    # Create one sample per concept
    for concept_name, concept_attributes in concept_to_attributes.items():
        # Create binary vector for this concept
        binary_attributes = [0] * len(all_attributes)
        for attr in concept_attributes:
            if attr in attribute_to_idx:
                idx = attribute_to_idx[attr]
                binary_attributes[idx] = 1

        # Add sample for this concept
        samples["concept"].append(concept_name)

        # Add binary value for each attribute
        for attr_name, attr_value in zip(all_attributes, binary_attributes):
            samples[attr_name].append(attr_value)

    logging.info(f"Created {len(samples['concept'])} concept-only dataset samples")
    logging.info(f"Concept dataset has {len(all_attributes)} attributes")

    return samples


def create_dataset(
    concept_file="dataset/mcrae-x-things.json",
    attribute_file="dataset/mcrae-x-things-taxonomy.json",
    image_dir="/home/bzq999/data/compmech/image_database_things/object_images",
    output_dir="/home/bzq999/data/compmech/mcrae-x-things.hf",
    concept_only_output_dir="/home/bzq999/data/compmech/mcrae-x-things-concepts-only.hf",
):
    """
    Create HuggingFace datasets from the concept-attribute data and images.
    Creates both an image-based dataset and a concept-only dataset in parallel.

    Args:
        concept_file: Path to JSON file with concept-attribute pairs
        attribute_file: Path to JSON file with attribute taxonomy
        image_dir: Path to directory containing object folders with images
        output_dir: Directory to save the image-based HuggingFace dataset
        concept_only_output_dir: Directory to save the concept-only HuggingFace dataset

    Returns:
        Tuple of (image_dataset, concept_only_dataset)
    """

    logging.info(f"Creating HuggingFace datasets from:")
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

    # Create image-based dataset samples
    logging.info("Creating image-based dataset...")
    dataset_samples = create_dataset_samples(
        object_to_images, concept_to_attributes, all_attributes
    )

    # Create concept-only dataset samples
    logging.info("Creating concept-only dataset...")
    concept_only_samples = create_concept_only_samples(
        concept_to_attributes, all_attributes
    )

    # Create HuggingFace datasets
    dataset = Dataset.from_dict(dataset_samples)
    concept_only_dataset = Dataset.from_dict(concept_only_samples)

    logging.info(f"Created image dataset with {len(dataset)} samples")
    logging.info(
        f"Created concept-only dataset with {len(concept_only_dataset)} samples"
    )
    logging.info(f"Image dataset features: {dataset.features}")
    logging.info(f"Concept-only dataset features: {concept_only_dataset.features}")

    # Save the image-based dataset
    output_path = Path(output_dir)
    dataset.save_to_disk(str(output_path))
    logging.info(f"Image dataset saved to: {output_path}")

    # Save the concept-only dataset
    concept_only_output_path = Path(concept_only_output_dir)
    concept_only_dataset.save_to_disk(str(concept_only_output_path))
    logging.info(f"Concept-only dataset saved to: {concept_only_output_path}")

    # Save metadata for image dataset
    metadata = {
        "dataset_name": output_path.stem,
        "dataset_type": "image_based",
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
    logging.info(f"Image dataset metadata saved to: {metadata_path}")

    # Save metadata for concept-only dataset
    concept_metadata = {
        "dataset_name": concept_only_output_path.stem,
        "dataset_type": "concept_only",
        "num_samples": len(concept_only_dataset),
        "num_attributes": num_attributes,
        "attribute_names": all_attributes,
        "num_objects": len(concept_to_attributes),
        "concept_file": concept_file,
        "attribute_file": attribute_file,
    }

    concept_metadata_path = concept_only_output_path / "metadata.json"
    with open(concept_metadata_path, "w") as f:
        json.dump(concept_metadata, f, indent=2)
    logging.info(f"Concept-only dataset metadata saved to: {concept_metadata_path}")

    return dataset, concept_only_dataset


if __name__ == "__main__":
    # Create both datasets
    image_dataset, concept_only_dataset = create_dataset(
        concept_file="dataset/mcrae-x-things.json",
        attribute_file="dataset/mcrae-x-things-taxonomy.json",
        image_dir="/home/bzq999/data/compmech/image_database_things/object_images",
    )
