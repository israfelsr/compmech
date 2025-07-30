#!/usr/bin/env python3
"""
Probe training script that loads features and trains logistic probes.
"""

import yaml
import logging
import argparse
import numpy as np
from pathlib import Path
from datasets import load_from_disk
import sys
import os
import json
from typing import List, Dict, Any

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.models.feature_extractors import get_feature_extractor
from src.models.probes import AttributeProbes


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("probe_training.log"),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train probes on extracted features")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--dataset_path", type=str, help="Override dataset path from config"
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["logistic", "linear", "mlp"],
        help="Override dataset path from config",
    )
    parser.add_argument("--layer", type=int, help="Override layer")
    parser.add_argument("--cv-folds", type=int, help="Override cross-validation folds")
    parser.add_argument("--n-repeats", type=int, help="Override of CV repeats")
    parser.add_argument("--output_dir", type=str, help="Override directory for results")
    parser.add_argument(
        "--specific-attributes",
        nargs="+",
        type=int,
        help="Train probes only for specific attribute indices",
    )

    args = parser.parse_args()

    setup_logging()

    # Load configuration
    config = load_config(args.config)
    logging.info(f"Loaded configuration from {args.config}")

    # Override paths if provided
    dataset_path = args.dataset_path or config["dataset"]["path"]

    dataset = load_from_disk(dataset_path)

    model_config = config["model"]
    layer_idx = model_config["layers"] or args.layer
    logging.info(f"Will probe layer: {layer_idx}")

    # Get layer features
    model_config = config["model"]
    extractor = get_feature_extractor(
        extractor_type=model_config["type"],
        model_name=model_config["model_name"],
        model_path=model_config["model_path"],
        batch_size=model_config["batch_size"],
        device=model_config["device"],
    )

    dataset = extractor.add_features_to_dataset(
        dataset=dataset,
        layers=layer_idx,
        features_dir=model_config["features_dir"],
    )

    # prepare features and labels
    # features = dataset["concept", f"layer_{layer_idx}"]
    # labels = dataset.remove_columns(["image_path", f"layer_{layer_idx}"])

    # Initialize probe trainer
    probe_config = config["probe"]
    probe_trainer = AttributeProbes(
        dataset=dataset,
        layer=f"layer_{layer_idx}",
        probe_type=probe_config["type"],
        random_seed=probe_config["seed"],
    )

    # Train probes
    if probe_config["specific_attribute"]:
        logging.info(
            f"Training probes for specific attributes: {probe_config['specific_attribute']}"
        )
        results = probe_trainer.evaluate_specific_attributes(
            attributes=probe_config["specific_attribute"],
            cv_folds=probe_config["cv_folds"],
            n_repeats=probe_config["n_repeats"],
        )
    else:
        logging.info("Training probes for all attributes")
        results = probe_trainer.train_all_probes(
            cv_folds=probe_config["cv_folds"],
            n_repeats=probe_config["n_repeats"],
        )

    # Save results
    if args.output_dir:
        output_dir = Path(probe_config["output_dir"]) or args.output_dir
    else:
        output_dir = Path("results/probes")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create results filename
    layers_str = "_".join([col.replace("layer_", "") for col in layer_idx])
    results_filename = f"probe_results_{probe_config['type']}_{layers_str}.json"
    results_path = output_dir / results_filename

    # Save results to JSON
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logging.info(f"Results saved to {results_path}")

    # Print summary
    print(f"\nProbe Training Summary:")
    print(f"- Probe type: {args.probe_type}")
    print(f"- Layers used: {layer_idx}")
    print(f"- Feature dimension: {features.shape[1]}")
    print(f"- Number of samples: {features.shape[0]}")
    print(f"- Number of attributes tested: {results['summary']['n_attributes_tested']}")
    print(
        f"- Mean F1 score: {results['summary']['mean_f1_across_attributes']:.4f} � {results['summary']['std_f1_across_attributes']:.4f}"
    )
    print(
        f"- Mean accuracy: {results['summary']['mean_accuracy_across_attributes']:.4f} � {results['summary']['std_accuracy_across_attributes']:.4f}"
    )
    print(f"- Results saved to: {results_path}")


if __name__ == "__main__":
    main()
