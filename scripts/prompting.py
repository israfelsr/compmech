import yaml
import json
import argparse
import logging
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
from tqdm import tqdm
import sys
import os

# Add src to path to import utilities
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.utils.logging_utils import setup_logging


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(args):
    # Load configuration
    config = load_config(args.config)
    logging.info(f"Loaded configuration from {args.config}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load dataset
    dataset_path = args.dataset_path or config["dataset"]["path"]
    dataset = load_from_disk(dataset_path)
    logging.info(f"Loaded dataset with {len(dataset)} samples")

    # Processing
    processor = AutoProcessor.from_pretrained(model_config["model_path"], use_fast=True)

    # Create prompt templates
    prompt_template = "USER: Image: <image>\nQuestion: Regarding the main object in the image, is the following statement true or false? The object has the attribute: '{attribute_name}'. Answer with only the word 'True' or 'False'.\nASSISTANT:"

    # Get attribute(s) from config or args
    specific_attribute = args.attribute or config["probe"].get("specific_attribute")

    if specific_attribute:
        # Single attribute mode
        attributes_to_process = specific_attribute
        logging.info(f"Processing single attribute: {specific_attribute}")
    else:
        # All attributes mode
        attributes_to_process = list(dataset[0].keys())[2:]
        logging.info(f"Processing all {len(attributes_to_process)} attributes")

    def preprocess_images(
        examples,
    ):
        """
        Loads images from paths and processes them using the given processor.
        This function is designed to be mapped over a HuggingFace Dataset.
        """
        image_paths = examples["image_path"]

        images = [Image.open(path).convert("RGB") for path in image_paths]
        text = ["<image>"] * len(images)
        inputs = processor(images=images, text=text, return_tensors="pt")
        result = {"image_path": image_paths}
        for key, value in inputs.items():
            result[key] = value
        return result

    image_dataset = dataset.map(
        preprocess_images,
        batched=True,
        load_from_cache_file=True,
        desc="Preprocessing Images",
    )

    def process_text(examples):
        image = Image.open(dataset[0]["image_path"]).convert("RGB")
        texts = [
            prompt_template.format(attribute_name=attr)
            for attr in attributes_to_process
        ]
        inputs = processor(images=image, text=texts, return_tensors="pt")
        for key, value in inputs.items():
            result[key] = value
        return result

    # Load model and processor
    model_config = config["model"]
    model = AutoModelForImageTextToText.from_pretrained(
        model_config["model_path"],
        torch_dtype=torch.float16,
        device_map=model_config["device"],
    )

    def process_sample_for_attribute(sample, attribute_name):
        """Process a single sample with image and prompt for a specific attribute."""
        image_path = sample["image_path"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Create prompt based on attribute
        prompt = prompt_template.format(attribute_name=attribute_name)

        # Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=100, do_sample=False
            )
            response = processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )[0]

        return {
            "image_path": image_path,
            "attribute": attribute_name,
            "prompt": prompt,
            "response": response.strip(),
            "label": sample[attribute_name],
        }

    # Process all samples for each attribute
    all_results = {}
    total_combinations = len(dataset) * len(attributes_to_process)
    logging.info(
        f"Processing {total_combinations} combinations ({len(dataset)} samples × {len(attributes_to_process)} attributes)..."
    )

    prompts = [
        prompt_template.format(attribute_name=attr) for attr in attributes_to_process
    ]

    processed_dataset = dataset.map

    for attribute_name in attributes_to_process:
        logging.info(f"Processing attribute: {attribute_name}")
        results = []

        for sample in tqdm(dataset, desc=f"Processing {attribute_name}"):
            try:
                result = process_sample_for_attribute(sample, attribute_name)
                results.append(result)
            except Exception as e:
                logging.error(
                    f"Error processing {sample.get('image_path', 'unknown')} for {attribute_name}: {e}"
                )
                continue

        all_results[attribute_name] = results
        logging.info(f"Completed {attribute_name}: {len(results)} samples processed")

    # Save results
    output_dir = Path(
        args.output_dir or config["probe"].get("output_dir", "results/prompting")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(attributes_to_process) == 1:
        # Single attribute: save as before
        attribute_name = attributes_to_process[0]
        results = all_results[attribute_name]
        results_filename = f"prompting_results_{attribute_name}_{args.prompt_type}.json"
        results_path = output_dir / results_filename

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logging.info(f"Results saved to {results_path}")
        logging.info(f"Processed {len(results)} samples successfully")
    else:
        # Multiple attributes: save separate files and a combined file
        total_samples = 0
        for attribute_name, results in all_results.items():
            results_filename = f"prompting_results_{attribute_name}.json"
            results_path = output_dir / results_filename

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            total_samples += len(results)
            logging.info(
                f"Saved {attribute_name}: {len(results)} samples -> {results_path}"
            )

        # Save combined results
        combined_filename = f"prompting_results_all_{args.prompt_type}.json"
        combined_path = output_dir / combined_filename

        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logging.info(f"Combined results saved to {combined_path}")
        logging.info(
            f"Processed {len(attributes_to_process)} attributes with {total_samples} total samples"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prompt LLAVA model with images and save responses"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--dataset-path", type=str, help="Override dataset path from config"
    )
    parser.add_argument(
        "--attribute",
        type=str,
        help="Specific attribute to prompt about. If not provided, processes all attributes in dataset",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["default", "taxonomic", "visual", "functional"],
        default="default",
        help="Type of prompt template to use",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    main(args)
