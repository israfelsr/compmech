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
        "--features_dir", type=str, help="Override output path from config"
    )

    args = parser.parse_args()

    setup_logging()

    # Load configuration
    config = load_config(args.config)
    logging.info(f"Loaded configuration from {args.config}")

    # Override paths if provided
    dataset_path = args.dataset_path or config["dataset"]["path"]
    features_dir = args.dataset_path or config["model"]["features_dir"]

    # Create output directory
    os.makedirs(features_dir, exist_ok=True)

    # Load dataset
    dataset = load_from_disk(dataset_path)
    logging.info(f"Loaded dataset with {len(dataset)} samples")

    # Initialize feature extractor
    model_config = config["model"]
    extractor = get_feature_extractor(
        extractor_type=model_config["type"],
        model_name=model_config["model_name"],
        model_path=model_config["model_path"],
        batch_size=model_config["batch_size"],
        device=model_config["device"],
        tower_name=model_config["tower_name"],
        projection_name=model_config["projection_name"],
    )

    logging.info(f"Initialized {model_config['type']} feature extractor")

    # Extract features and add to dataset
    dataset_with_features = extractor.extract_and_save(
        dataset=dataset,
        features_dir=model_config["features_dir"],
    )

    logging.info("Feature extraction completed successfully!")


if __name__ == "__main__":
    main()
