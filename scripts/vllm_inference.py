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

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path to import utilities
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.utils.logging_utils import setup_logging

# VLLM imports
from vllm import LLM, SamplingParams


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_vllm_prompts_batch(image_paths: list, attribute_name: str) -> list:
    """Create VLLM-compatible prompts for a batch of images and a single attribute."""
    question = f"Regarding the main object in the image, is the following statement true or false? The object has the attribute: '{attribute_name}'. Answer with only the word 'True' or 'False'."
    
    # Qwen2.5-VL format according to official VLLM example
    prompt_text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    vllm_prompts = []
    for image_path in image_paths:
        vllm_prompt = {
            "prompt": prompt_text,
            "multi_modal_data": {"image": Image.open(image_path).convert("RGB")},
        }
        vllm_prompts.append(vllm_prompt)

    return vllm_prompts


def process_samples_for_attribute(
    llm, sampling_params, dataset, attribute_name, batch_size
):
    """Process all samples for a specific attribute using batched VLLM inference."""
    results = []

    # Process in batches
    for i in tqdm(
        range(0, len(dataset), batch_size), desc=f"Processing {attribute_name} batches"
    ):
        batch_end = min(i + batch_size, len(dataset))
        batch_samples = dataset[i:batch_end]

        # Extract image paths from batch
        image_paths = batch_samples["image_path"]

        # Create VLLM prompts for this batch and attribute
        vllm_prompts = create_vllm_prompts_batch(image_paths, attribute_name)

        try:
            # Generate responses using VLLM
            outputs = llm.generate(vllm_prompts, sampling_params)

            # Process results for this batch
            for j, (sample, output) in enumerate(zip(batch_samples, outputs)):
                response = output.outputs[0].text.strip()

                results.append(
                    {
                        "image_path": sample["image_path"],
                        "attribute": attribute_name,
                        "prompt": vllm_prompts[j]["prompt"],
                        "response": response,
                        "label": sample.get(attribute_name, None),
                    }
                )

        except Exception as e:
            logging.error(
                f"Error processing batch {i}-{batch_end} for attribute {attribute_name}: {e}"
            )
            # Add empty results for failed batch
            for sample in batch_samples:
                results.append(
                    {
                        "image_path": sample["image_path"],
                        "attribute": attribute_name,
                        "prompt": "",
                        "response": "",
                        "label": sample.get(attribute_name, None),
                    }
                )

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

    # Initialize VLLM model with Qwen2.5-VL
    logging.info(f"Loading VLLM model: {model_path}")
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        limit_mm_per_prompt={"image": 1},
    )

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        stop_token_ids=None,
    )

    # Get batch size from config
    batch_size = model_config.get("batch_size", args.batch_size)

    # Process all samples for each attribute
    all_results = {}

    for attr_idx, attribute_name in enumerate(attributes_to_process):
        logging.info(
            f"Processing attribute {attr_idx+1}/{len(attributes_to_process)}: {attribute_name}"
        )

        try:
            results = process_samples_for_attribute(
                llm, sampling_params, dataset, attribute_name, batch_size
            )
            all_results[attribute_name] = results
            logging.info(
                f"Completed {attribute_name}: {len(results)} samples processed"
            )
        except Exception as e:
            logging.error(f"Error processing attribute {attribute_name}: {e}")
            all_results[attribute_name] = []  # Empty results for failed attribute
            continue

    # Save results
    output_dir = Path(
        args.output_dir or config["probe"].get("output_dir", "results/vllm_inference")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(attributes_to_process) == 1:
        # Single attribute: save as before
        attribute_name = attributes_to_process[0]
        results = all_results[attribute_name]
        results_filename = f"vllm_inference_results_{attribute_name}.json"
        results_path = output_dir / results_filename

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logging.info(f"Results saved to {results_path}")
        logging.info(f"Processed {len(results)} samples successfully")
    else:
        # Multiple attributes: save separate files and a combined file
        total_samples = 0
        for attribute_name, results in all_results.items():
            results_filename = f"vllm_inference_results_{attribute_name}.json"
            results_path = output_dir / results_filename

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            total_samples += len(results)
            logging.info(
                f"Saved {attribute_name}: {len(results)} samples -> {results_path}"
            )

        # Save combined results
        combined_filename = f"vllm_inference_results_all.json"
        combined_path = output_dir / combined_filename

        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logging.info(f"Combined results saved to {combined_path}")
        logging.info(
            f"Processed {len(attributes_to_process)} attributes with {total_samples} total samples"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with Qwen2.5-VL using VLLM")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--attribute",
        type=str,
        help="Specific attribute to prompt about. If not provided, processes all attributes in dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Modify batch size from config",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    main(args)
