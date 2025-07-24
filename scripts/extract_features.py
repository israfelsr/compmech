#!/usr/bin/env python3
"""
Feature extraction script using DINOv2FeatureExtractor.
Extracts features from images in a dataset and saves them.
"""

import yaml
import logging
import argparse
from pathlib import Path
from datasets import load_from_disk
import sys
import os

# Add src to path to import feature extractors
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.models.feature_extractors import get_feature_extractor


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("feature_extraction.log"),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from dataset using DINOv2"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--dataset-path", type=str, help="Override dataset path from config"
    )
    parser.add_argument(
        "--output-path", type=str, help="Override output path from config"
    )

    args = parser.parse_args()

    setup_logging()

    # Load configuration
    config = load_config(args.config)
    logging.info(f"Loaded configuration from {args.config}")

    # Override paths if provided
    dataset_path = args.dataset_path or config["dataset"]["path"]
    output_path = args.output_path or config["dataset"]["output_path"]

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(config["extraction"]["features_base_dir"], exist_ok=True)

    # Load dataset
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    logging.info(f"Loaded dataset with {len(dataset)} samples")

    # Initialize feature extractor
    model_config = config["model"]
    extractor = get_feature_extractor(
        extractor_type=model_config["type"],
        model_name=model_config["model_name"],
        model_path=model_config["model_path"],
        layer=model_config["layer"],
        batch_size=model_config["batch_size"],
        device=model_config["device"],
    )

    logging.info(f"Initialized {model_config['type']} feature extractor")

    # Extract features and add to dataset
    extraction_config = config["extraction"]
    dataset_with_features = extractor.add_features_to_dataset(
        dataset=dataset,
        layers=extraction_config["layers"],
        batch_size=model_config["batch_size"],
        features_base_dir=extraction_config["features_base_dir"],
    )

    # Save dataset with features
    logging.info(f"Saving dataset with features to {output_path}")
    dataset_with_features.save_to_disk(output_path)

    logging.info("Feature extraction completed successfully!")

    # Print summary
    print(f"\nExtraction Summary:")
    print(f"- Original dataset size: {len(dataset)}")
    print(f"- Features extracted from layers: {extraction_config['layers']}")
    print(f"- Model used: {model_config['model_path']}")
    print(f"- Dataset with features saved to: {output_path}")


if __name__ == "__main__":
    main()
