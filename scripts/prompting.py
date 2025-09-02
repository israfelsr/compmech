import yaml
import json
import argparse
import logging
from pathlib import Path
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
import sys
import os
import torch

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path to import utilities
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logging_utils import setup_logging

# Transformers imports
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    AutoModelForImageTextToText,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_and_processor(model_path: str, model_type: str):
    """Load model and processor based on model type."""
    if "paligemma2" in model_type:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        processor = PaliGemmaProcessor.from_pretrained(model_path, use_fast=True)
        return model, processor
    else:
        # Generic fallback for other vision-language models
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        return model, processor


def create_prompt(attribute_name: str, model_type: str) -> str:
    """Create model-specific prompt for attribute classification."""
    question = f"Regarding the main object in the image, is the following statement true or false? The object has the attribute: '{attribute_name}'. Answer with only the word 'True' or 'False'."

    if "paligemma2" in model_type:
        return f"answer en {question}"
    elif "qwen2.5-vl" in model_type.lower():
        # Qwen2.5-VL format
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # Default format
        return question


def process_single_sample(
    model,
    processor,
    image_path: str,
    attribute_name: str,
    label: int,
    model_type: str,
    max_tokens: int = 10,
) -> dict:
    """Process a single image-attribute pair."""
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Create prompt
        prompt = create_prompt(attribute_name, model_type)

        # Process inputs
        model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(
            model.device
        )

        # Convert to appropriate dtype
        if hasattr(model_inputs, "pixel_values"):
            model_inputs["pixel_values"] = model_inputs["pixel_values"].to(
                torch.bfloat16
            )

        input_len = model_inputs["input_ids"].shape[-1]

        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True).strip()

        return {
            "image_path": image_path,
            "attribute": attribute_name,
            "prompt": prompt,
            "response": decoded,
            "label": label,
        }

    except Exception as e:
        logging.error(
            f"Error processing {image_path} with attribute {attribute_name}: {e}"
        )
        return {
            "image_path": image_path,
            "attribute": attribute_name,
            "prompt": prompt if "prompt" in locals() else "",
            "response": "",
            "label": label,
        }


def process_samples_for_attribute(
    model,
    processor,
    dataset,
    attribute_name: str,
    model_type: str,
    max_tokens: int = 10,
):
    """Process all samples for a specific attribute."""
    results = []

    for sample in tqdm(dataset, desc=f"Processing {attribute_name}"):
        result = process_single_sample(
            model,
            processor,
            sample["image_path"],
            attribute_name,
            sample[attribute_name],
            model_type,
            max_tokens,
        )
        results.append(result)

    return results


def main(args):
    # Load configuration
    config = load_config(args.config)
    logging.info(f"Loaded configuration from {args.config}")

    # Load dataset
    dataset_path = config["dataset"]["path"]
    dataset = load_from_disk(dataset_path)
    logging.info(f"Loaded dataset with {len(dataset)} samples")

    # Get model configuration
    model_config = config["model"]
    model_path = model_config["model_path"]
    model_type = model_config.get("type", "unknown")
    max_tokens = model_config.get("max_tokens", 10)

    # Load model and processor
    logging.info(f"Loading model: {model_path}")
    model, processor = get_model_and_processor(model_path, model_type)
    logging.info(f"Model loaded successfully on device: {model.device}")

    # Get attribute(s) from config or args
    specific_attribute = args.attribute or config["probe"].get("specific_attribute")

    if specific_attribute:
        # Single attribute mode
        if isinstance(specific_attribute, str):
            attributes_to_process = [specific_attribute]
        else:
            attributes_to_process = specific_attribute
        logging.info(f"Processing specific attributes: {attributes_to_process}")
    else:
        # All attributes mode
        attributes_to_process = list(dataset[0].keys())[
            2:
        ]  # Skip image_path and any metadata
        logging.info(f"Processing all {len(attributes_to_process)} attributes")

    # Set up output directory
    output_dir = Path(
        args.output_dir or config["probe"].get("output_dir", "results/hf_inference")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all samples for each attribute
    all_results = {}

    for attr_idx, attribute_name in enumerate(attributes_to_process):
        logging.info(
            f"Processing attribute {attr_idx+1}/{len(attributes_to_process)}: {attribute_name}"
        )

        try:
            results = process_samples_for_attribute(
                model, processor, dataset, attribute_name, model_type, max_tokens
            )
            all_results[attribute_name] = results

            # Save individual attribute results immediately
            results_filename = f"hf_inference_results_{attribute_name}.json"
            results_path = output_dir / results_filename

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logging.info(
                f"Completed {attribute_name}: {len(results)} samples processed -> saved to {results_path}"
            )

        except Exception as e:
            logging.error(f"Error processing attribute {attribute_name}: {e}")
            continue

    # Save combined results if multiple attributes
    if len(attributes_to_process) > 1:
        combined_filename = f"hf_inference_results_all.json"
        combined_path = output_dir / combined_filename

        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        total_samples = sum(len(results) for results in all_results.values())
        logging.info(f"Combined results saved to {combined_path}")
        logging.info(
            f"Processed {len(attributes_to_process)} attributes with {total_samples} total samples"
        )
    else:
        # Single attribute case
        if attributes_to_process[0] in all_results:
            total_samples = len(all_results[attributes_to_process[0]])
            logging.info(f"Processed {total_samples} samples successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with Vision-Language models using Hugging Face Transformers"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--attribute",
        type=str,
        help="Specific attribute to prompt about. If not provided, processes all attributes in dataset",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    main(args)
