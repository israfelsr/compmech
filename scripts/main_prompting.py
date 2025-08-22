#!/usr/bin/env python3
"""
Main prompting script using model abstraction for VLMs.
Supports both VLLM and HuggingFace models for attribute prompting.
"""

import yaml
import json
import argparse
import logging
from pathlib import Path
from datasets import load_from_disk
import torch
from PIL import Image
from tqdm import tqdm
import sys
import os
import random
import numpy as np
from vllm import LLM, SamplingParams
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    GenerationConfig,
)


# Add src to path to import utilities
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.utils.logging_utils import setup_logging

from model_utils import query_model

TEMPERATURE = 0.0  # Use deterministic generation for attribute testing
MAX_TOKENS = 50  # Short answers for True/False questions

SUPPORTED_MODELS = [
    "llava",
    "qwen2.5-7b",
    "qwen2.5-3b",
    "qwen2.5-32b",
    "qwen2.5-72b",
]

# Model type mapping
VLLM_MODELS = [
    "qwen2.5-7b",
    "qwen2.5-3b",
    "qwen2.5-32b",
    "qwen2.5-72b",
]

HUGGINGFACE_MODELS = ["llava"]


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_prompt_for_model(
    model_name: str, attribute_name: str, prompt_type: str = "default"
):
    """Generate model-specific prompt format."""
    base_prompt = "USER: Image: <image>\nQuestion: Regarding the main object in the image, is the following statement true or false? The object has the attribute: '{attribute_name}'. Answer with only the word 'True' or 'False'.\nASSISTANT:"

    if model_name in ["qwen2.5-7b", "qwen2.5-3b", "qwen2.5-32b", "qwen2.5-72b"]:
        # VLLM format for Qwen models
        return [{"role": "user", "content": base_prompt}]
    elif model_name == "llava" or "llava" in model_name.lower():
        # LLaVA format
        return base_prompt
    else:
        # Default format
        return base_prompt


class AttributePromptingPipeline:
    """Pipeline for attribute prompting with different VLM models."""

    def __init__(
        self, model_name: str, model_path: str, device: str = "cuda", ngpu: int = 1
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.ngpu = ngpu

        # Initialize model
        self.model, self.processor = self.initialize_model(
            model_name=model_name,
            model_path=model_path,
            device=device,
            ngpu=ngpu,
        )

        logging.info(f"Initialized {model_name} model from {model_path}")

    def initialize_model(
        self,
        model_name: str,
        model_path: str,
        device: str = "cuda",
        ngpu: int = 1,
    ):
        """
        Initialize the model and processor based on the model name.
        """
        if model_name not in SUPPORTED_MODELS:
            raise NotImplementedError(
                f"Model {model_name} not supported. Supported models: {SUPPORTED_MODELS}"
            )

        if model_name in VLLM_MODELS:
            if not VLLM_AVAILABLE:
                raise ImportError(
                    "VLLM is not installed. Please install it to use VLLM models."
                )

            logging.info(f"Initializing {model_name} with VLLM")

            if model_name in ["qwen2.5-7b", "qwen2.5-3b", "qwen2.5-32b", "qwen2.5-72b"]:
                model = LLM(
                    model_path,
                    tensor_parallel_size=ngpu,
                    max_model_len=4096,  # Shorter for attribute testing
                    dtype="bfloat16",
                )
                processor = AutoProcessor.from_pretrained(
                    model_path, use_fast=True, trust_remote_code=True
                )

        elif model_name in HUGGINGFACE_MODELS:
            logging.info(f"Initializing {model_name} with HuggingFace Transformers")

            if model_name == "llava" or "llava" in model_name.lower():
                model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map=device if device != "cpu" else None,
                    trust_remote_code=True,
                )
                processor = AutoProcessor.from_pretrained(
                    model_path, use_fast=True, trust_remote_code=True
                )

        else:
            raise ValueError(f"Unknown model type for {model_name}")

        logging.info(f"Model {model_name} loaded from {model_path}")
        return model, processor

    def process_sample_for_attribute(
        self, sample: dict, attribute_name: str, prompt_type: str = "default"
    ):
        """Process a single sample for a specific attribute."""
        image_path = sample["image_path"]

        # Generate prompt
        prompt = generate_prompt_for_model(self.model_name, attribute_name, prompt_type)

        # Prepare image paths (some models need this)
        if os.path.exists(image_path):
            image_paths = [image_path]
        else:
            logging.warning(f"Image not found: {image_path}")
            image_paths = None

        try:
            # Query model
            response, reasoning = query_model(
                model_name=self.model_name,
                model=self.model,
                processor=self.processor,
                prompt=prompt,
                images=image_paths,
                device=self.device,
            )

            return {
                "image_path": image_path,
                "attribute": attribute_name,
                "prompt": str(prompt),
                "response": response.strip() if response else "",
                "reasoning": reasoning if reasoning else "",
                "label": sample.get(attribute_name, None),
                "model": self.model_name,
            }

        except Exception as e:
            logging.error(f"Error processing {image_path} for {attribute_name}: {e}")
            return {
                "image_path": image_path,
                "attribute": attribute_name,
                "prompt": str(prompt),
                "response": f"ERROR: {str(e)}",
                "reasoning": "",
                "label": sample.get(attribute_name, None),
                "model": self.model_name,
            }

    def process_batch_for_image(
        self, sample: dict, attributes: list, prompt_type: str = "default"
    ):
        """Process multiple attributes for a single image."""
        results = []

        for attribute_name in attributes:
            result = self.process_sample_for_attribute(
                sample, attribute_name, prompt_type
            )
            results.append(result)

        return results


def main(args):
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load configuration
    config = load_config(args.config)
    logging.info(f"Loaded configuration from {args.config}")

    # Load dataset
    dataset_path = args.dataset_path or config["dataset"]["path"]
    dataset = load_from_disk(dataset_path)
    logging.info(f"Loaded dataset with {len(dataset)} samples")

    # Limit samples if specified
    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        logging.info(f"Limited to {len(dataset)} samples")

    # Initialize prompting pipeline
    model_config = config["model"]
    pipeline = AttributePromptingPipeline(
        model_name=args.model or model_config.get("model_name", "llava"),
        model_path=args.model_path or model_config["model_path"],
        device=args.device or "cuda",
        ngpu=args.ngpu,
    )

    # Get attribute(s) from config or args
    specific_attribute = args.attribute or config["probe"].get("specific_attribute")

    if specific_attribute:
        # Single attribute mode
        attributes_to_process = [specific_attribute]
        logging.info(f"Processing single attribute: {specific_attribute}")
    else:
        # All attributes mode
        attributes_to_process = list(dataset[0].keys())[2:]
        logging.info(f"Processing all {len(attributes_to_process)} attributes")

    # Process dataset
    all_results = {}
    total_combinations = len(dataset) * len(attributes_to_process)
    logging.info(
        f"Processing {total_combinations} combinations ({len(dataset)} samples × {len(attributes_to_process)} attributes)..."
    )

    if args.batch_per_image:
        # Batch processing: all attributes per image
        for sample in tqdm(dataset, desc="Processing images"):
            try:
                batch_results = pipeline.process_batch_for_image(
                    sample, attributes_to_process, args.prompt_type
                )

                # Organize results by attribute
                for result in batch_results:
                    attribute_name = result["attribute"]
                    if attribute_name not in all_results:
                        all_results[attribute_name] = []
                    all_results[attribute_name].append(result)

            except Exception as e:
                logging.error(
                    f"Error processing {sample.get('image_path', 'unknown')}: {e}"
                )
                continue
    else:
        # Sequential processing: one attribute at a time
        for attribute_name in attributes_to_process:
            logging.info(f"Processing attribute: {attribute_name}")
            results = []

            for sample in tqdm(dataset, desc=f"Processing {attribute_name}"):
                try:
                    result = pipeline.process_sample_for_attribute(
                        sample, attribute_name, args.prompt_type
                    )
                    results.append(result)
                except Exception as e:
                    logging.error(
                        f"Error processing {sample.get('image_path', 'unknown')} for {attribute_name}: {e}"
                    )
                    continue

            all_results[attribute_name] = results
            logging.info(
                f"Completed {attribute_name}: {len(results)} samples processed"
            )

    # Save results
    output_dir = Path(
        args.output_dir or config["probe"].get("output_dir", "results/main_prompting")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual attribute results
    total_processed = 0
    for attribute_name, results in all_results.items():
        results_filename = f"main_prompting_results_{args.model}_{attribute_name}_{args.prompt_type}.json"
        results_path = output_dir / results_filename

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        total_processed += len(results)
        logging.info(
            f"Saved {attribute_name}: {len(results)} samples -> {results_path}"
        )

    # Save combined results
    if len(attributes_to_process) > 1:
        combined_filename = (
            f"main_prompting_results_{args.model}_all_{args.prompt_type}.json"
        )
        combined_path = output_dir / combined_filename

        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logging.info(f"Combined results saved to {combined_path}")

    logging.info(
        f"Processing complete! {len(attributes_to_process)} attributes, {total_processed} total samples"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prompt VLM models for attribute testing with model abstraction"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--dataset-path", type=str, help="Override dataset path from config"
    )
    parser.add_argument(
        "--model", type=str, help="Model name (e.g., 'llava', 'qwen2.5-7b', 'molmo')"
    )
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--attribute",
        type=str,
        help="Specific attribute to test. If not provided, processes all attributes in dataset",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["default", "descriptive", "comparative"],
        default="default",
        help="Type of prompt to use for attribute testing",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument(
        "--num-samples", type=int, help="Limit number of samples to process"
    )
    parser.add_argument(
        "--batch-per-image",
        action="store_true",
        help="Process all attributes per image in batch (may be more efficient)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for inference"
    )
    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    main(args)
