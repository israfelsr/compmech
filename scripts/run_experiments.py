#!/usr/bin/env python3
"""
Main script to run attribute probing experiments.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiments.experiment_runner import ExperimentRunner
from utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run attribute probing experiments")

    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment_configs.yaml",
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        choices=["single", "comparison", "layer_analysis"],
        default="single",
        help="Type of experiment to run",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--wandb-project", type=str, default="attribute-probes", help="W&B project name"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of cached features",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="external_data/cached_features",
        help="Directory for feature caching",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Setup logging
    setup_logging()

    # Initialize experiment runner
    runner = ExperimentRunner(
        concept_file="external_data/raw/mcrae-x-things.json",
        attribute_file="external_data/raw/mcrae-x-things-taxonomy.json",
        image_dir="external_data/raw/images",
        results_dir=args.output_dir,
        cache_dir=args.cache_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )

    if args.experiment_type == "single":
        # Example single experiment
        extractor_config = {
            "extractor_type": "dinov2",
            "model_name": "facebook/dinov2-base",
            "layer": "last",
        }
        probe_config = {"probe_type": "logistic"}

        results = runner.run_single_experiment(
            extractor_config=extractor_config,
            probe_config=probe_config,
            experiment_name="dinov2_base_logistic",
            force_recompute_features=args.force_recompute,
        )

    elif args.experiment_type == "comparison":
        # Model comparison
        from experiments.comparative_studies import run_model_comparison

        results = run_model_comparison(runner)

    elif args.experiment_type == "layer_analysis":
        # Layer analysis
        from experiments.comparative_studies import run_layer_analysis

        results = run_layer_analysis(runner)

    logging.info("Experiment completed successfully!")
    return results


if __name__ == "__main__":
    main()
