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

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path to import utilities
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.utils.logging_utils import setup_logging


class LlavaPrompting:
    def __init__(self, model_path, device, batch_size, processor):
        self.model = AutoModelForImageTextToText.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.processor = processor

    def inference(self, pixel_values, text):
        with torch.no_grad():
            pixel_values = pixel_values[None].to(self.device)
            vision_feature_layer = self.model.config.vision_feature_layer
            vision_feature_select_strategy = (
                self.model.config.vision_feature_select_strategy
            )
            image_features = self.model.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            for i in range(0, text.input_ids.shape[0], self.batch_size):
                input_ids = text["input_ids"][i : i + self.batch_size, :].to(
                    self.device
                )
                attention_mask = text["attention_mask"][i : i + self.batch_size, :].to(
                    self.device
                )
                real_batch_size = input_ids.shape[0]

                inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
                image_features_batch = (
                    image_features[0]
                    .repeat(self.batch_size, 1)
                    .to(inputs_embeds.device, inputs_embeds.dtype)
                )
                special_image_mask = self.model.model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_features_batch,
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_image_mask, image_features_batch
                )
                generated_ids = self.model.generate(
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=100,
                    do_sample=False,
                )
                responses = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                return responses


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint data if it exists."""
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        logging.info(f"Loaded {len(checkpoint.get('results', []))} processed items from checkpoint")
        return checkpoint
    else:
        logging.info("No checkpoint found, starting from beginning")
        return {"results": [], "processed_combinations": set()}


def save_checkpoint(checkpoint_path: str, results: list, processed_combinations: set):
    """Save current progress to checkpoint file."""
    checkpoint = {
        "results": results,
        "processed_combinations": list(processed_combinations),
        "timestamp": str(torch.tensor(0).item()),  # Simple timestamp
        "total_processed": len(results)
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    logging.info(f"Checkpoint saved with {len(results)} processed items")


def create_combination_key(sample_idx: int, attribute_name: str) -> str:
    """Create a unique key for sample-attribute combination."""
    return f"{sample_idx}_{attribute_name}"


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
    model_config = config["model"]
    processor = AutoProcessor.from_pretrained(model_config["model_path"], use_fast=True)

    # Create prompt template
    prompt_template = "USER: Image: <image>\nQuestion: Regarding the main object in the image, is the following statement true or false? The object has the attribute: '{attribute_name}'. Answer with only the word 'True' or 'False'.\nASSISTANT:"

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

    def preprocess_images(examples):
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

    # Preprocess images in the dataset
    image_dataset = dataset.map(
        preprocess_images,
        batched=True,
        load_from_cache_file=True,
        desc="Preprocessing Images",
        num_proc=1,
    )

    # Initialize checkpoint system
    checkpoint_path = args.checkpoint_file or os.path.join(
        args.output_dir or config["probe"].get("output_dir", "results/main_prompting"),
        "checkpoint.json"
    )
    
    # Load existing checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    all_results = checkpoint.get("results", [])
    processed_combinations = set(checkpoint.get("processed_combinations", []))
    
    logging.info(f"Starting with {len(all_results)} already processed items")
    logging.info(f"Will save checkpoint every 100 processed items to: {checkpoint_path}")

    # Initialize LlavaPrompting class
    batch_size = model_config.get("batch_size", 16)
    llava_model = LlavaPrompting(
        model_path=model_config["model_path"],
        device=device,
        batch_size=batch_size,
        processor=processor,
    )

    image_dataset.set_format(
        type="torch",
        columns=["pixel_values", "image_path"],
    )

    # Process each sample with all its attributes
    items_since_checkpoint = 0
    
    for sample_idx, sample in enumerate(tqdm(image_dataset, desc="Processing samples")):
        logging.info(f"Processing sample {sample_idx+1}/{len(image_dataset)}")

        # Check which attributes for this sample need processing
        attributes_to_process_sample = []
        for attribute_name in attributes_to_process:
            combination_key = create_combination_key(sample_idx, attribute_name)
            if combination_key not in processed_combinations:
                attributes_to_process_sample.append(attribute_name)
        
        # Skip this sample if all attributes are already processed
        if not attributes_to_process_sample:
            logging.info(f"Sample {sample_idx} already fully processed, skipping")
            continue

        # Prepare image data for this sample
        pixel_values = sample["pixel_values"]

        # Create prompts only for unprocessed attributes
        texts = [
            prompt_template.format(attribute_name=attr)
            for attr in attributes_to_process_sample
        ]

        # Process text prompts
        text_inputs = processor(
            images=Image.open(dataset[0]["image_path"]).convert("RGB"),
            text=texts,
            return_tensors="pt",
            padding=True,
        )

        try:
            # Use LlavaPrompting class for inference
            responses = llava_model.inference(pixel_values, text_inputs)

            # Process results for this sample
            for attr_idx, (attribute_name, response) in enumerate(
                zip(attributes_to_process_sample, responses)
            ):
                # Get original sample data for label
                original_sample = dataset[sample_idx]
                
                combination_key = create_combination_key(sample_idx, attribute_name)
                processed_combinations.add(combination_key)

                all_results.append(
                    {
                        "image_path": sample["image_path"],
                        "attribute": attribute_name,
                        "prompt": processor.decode(
                            text_inputs["input_ids"][attr_idx], skip_special_tokens=True
                        ),
                        "response": response.strip(),
                        "label": original_sample.get(attribute_name, None),
                    }
                )
                
                items_since_checkpoint += 1
                
                # Save checkpoint every 100 items
                if items_since_checkpoint >= 100:
                    save_checkpoint(checkpoint_path, all_results, processed_combinations)
                    items_since_checkpoint = 0

        except Exception as e:
            logging.error(f"Error processing sample {sample_idx}: {e}")
            # Add empty results for failed sample attributes that weren't processed yet
            for attribute_name in attributes_to_process_sample:
                combination_key = create_combination_key(sample_idx, attribute_name)
                processed_combinations.add(combination_key)
                
                all_results.append(
                    {
                        "image_path": sample["image_path"],
                        "attribute": attribute_name,
                        "prompt": "",
                        "response": "",
                        "label": dataset[sample_idx].get(attribute_name, None),
                    }
                )
                
                items_since_checkpoint += 1
                
                # Save checkpoint every 100 items (including errors)
                if items_since_checkpoint >= 100:
                    save_checkpoint(checkpoint_path, all_results, processed_combinations)
                    items_since_checkpoint = 0
            continue

    # Save final checkpoint
    save_checkpoint(checkpoint_path, all_results, processed_combinations)
    
    # Save final results
    output_dir = Path(
        args.output_dir or config["probe"].get("output_dir", "results/main_prompting")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_filename = f"main_prompting_results_{args.prompt_type}.json"
    results_path = output_dir / results_filename

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logging.info(f"Results saved to {results_path}")
    logging.info(f"Final checkpoint saved to {checkpoint_path}")
    logging.info(f"Processed {len(all_results)} total combinations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prompt LLAVA model with images using LlavaPrompting class"
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
    parser.add_argument(
        "--checkpoint-file", 
        type=str, 
        help="Path to checkpoint file for resuming progress. If not specified, uses 'checkpoint.json' in output directory"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    main(args)
