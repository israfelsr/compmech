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


class QwenPrompting:
    def __init__(self, model_path, device, batch_size, processor):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.processor = processor
        self.model = None

    def load_model(self):
        """Load the model when ready for inference."""
        if self.model is None:
            logging.info(f"Loading Qwen model from {self.model_path}")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path
            ).to(self.device)
            self.model.eval()
            logging.info("Model loaded successfully")

    def inference_sample_attributes(
        self, sample, attributes_to_process, prompt_template
    ):
        """Process all attributes for a single sample."""
        image = sample["images"]  # PIL image from preprocessing
        sample_idx = sample.get("sample_idx", 0)

        # Create messages for all attributes for this sample
        batch_messages = []
        batch_info = []

        for attribute_name in attributes_to_process:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": prompt_template.format(
                                attribute_name=attribute_name
                            ),
                        },
                    ],
                }
            ]
            batch_messages.append(messages)
            batch_info.append(
                {
                    "sample_idx": sample_idx,
                    "image_path": sample["image_path"],
                    "attribute": attribute_name,
                    "label": sample.get(attribute_name, None),
                }
            )

        # Process in mini-batches
        responses = []
        with torch.no_grad():
            for i in range(0, len(batch_messages), self.batch_size):
                mini_batch_messages = batch_messages[i : i + self.batch_size]
                mini_batch_info = batch_info[i : i + self.batch_size]

                # Process each message and prepare inputs
                batch_inputs_list = []
                for messages in mini_batch_messages:
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    batch_inputs_list.append(inputs)

                # Handle batching
                if len(batch_inputs_list) == 1:
                    batch_inputs = {
                        k: v.to(self.device) for k, v in batch_inputs_list[0].items()
                    }
                else:
                    batch_inputs = self._pad_and_stack_inputs(batch_inputs_list)

                # Generate responses
                generated_ids = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                    or self.processor.tokenizer.eos_token_id,
                )

                # Decode responses
                input_length = batch_inputs["input_ids"].shape[1]
                batch_responses = self.processor.batch_decode(
                    generated_ids[:, input_length:],
                    skip_special_tokens=True,
                )

                # Combine with sample info
                for info, response in zip(mini_batch_info, batch_responses):
                    responses.append(
                        {
                            "sample_idx": info["sample_idx"],
                            "image_path": info["image_path"],
                            "attribute": info["attribute"],
                            "response": response.strip(),
                            "label": info["label"],
                        }
                    )

        return responses

    def _pad_and_stack_inputs(self, batch_inputs_list):
        """Pad and stack inputs for batch processing."""
        batch_inputs = {}

        # Find max length for padding
        max_length = max(inp["input_ids"].shape[1] for inp in batch_inputs_list)

        for key in batch_inputs_list[0].keys():
            if key in ["input_ids", "attention_mask"]:
                # Pad sequences to same length
                padded = []
                for inp in batch_inputs_list:
                    seq = inp[key].squeeze(0)
                    if seq.shape[0] < max_length:
                        if key == "input_ids":
                            pad_value = (
                                self.processor.tokenizer.pad_token_id
                                or self.processor.tokenizer.eos_token_id
                            )
                        else:  # attention_mask
                            pad_value = 0
                        padding = torch.full(
                            (max_length - seq.shape[0],), pad_value, dtype=seq.dtype
                        )
                        seq = torch.cat([seq, padding])
                    padded.append(seq)
                batch_inputs[key] = torch.stack(padded).to(self.device)
            else:
                # For other keys like pixel_values, images, etc.
                batch_inputs[key] = torch.cat(
                    [inp[key] for inp in batch_inputs_list], dim=0
                ).to(self.device)

        return batch_inputs


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
        logging.info(
            f"Loaded {len(checkpoint.get('results', []))} processed items from checkpoint"
        )
        return checkpoint
    else:
        logging.info("No checkpoint found, starting from beginning")
        return {"results": [], "last_completed_idx": -1}


def save_checkpoint(checkpoint_path: str, results: list, last_completed_idx: int):
    """Save current progress to checkpoint file."""
    checkpoint = {
        "results": results,
        "last_completed_idx": last_completed_idx,
        "total_processed": len(results),
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    logging.info(f"Checkpoint saved with {len(results)} total results")


def main(args):
    config = load_config(args.config)
    logging.info(f"Loaded configuration from {args.config}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load dataset
    dataset = load_from_disk(config["dataset"]["path"])
    logging.info(f"Loaded dataset with {len(dataset)} samples")

    # Setup processor
    model_config = config["model"]
    processor = AutoProcessor.from_pretrained(model_config["model_path"], use_fast=True)
    logging.info("Processor loaded")

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

    # Create prompt template
    prompt_template = "Question: Regarding the main object in the image, is the following statement true or false? The object has the attribute: '{attribute_name}'. Answer with only the word 'True' or 'False'."

    def create_all_combinations(examples):
        images = [Image.open(p).convert("RGB") for p in examples["image_path"]]

        # Pre-allocate result structure
        result = {
            "image": [],
            "image_path": [],
            "attribute": [],
            "conversation": [],
            "label": [],
        }

        # Create all combinations
        for i, (img, img_path) in enumerate(zip(images, examples["image_path"])):
            for attr in attributes_to_process:
                # Create conversation for this combination
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {
                                "type": "text",
                                "text": prompt_template.format(attribute_name=attr),
                            },
                        ],
                    }
                ]

                result["image"].append(img)
                result["image_path"].append(img_path)
                result["attribute"].append(attr)
                result["conversation"].append(conversation)
                result["label"].append(examples.get(attr, [None] * len(images))[i])

        return result

    # Step 1: Create ALL combinations efficiently
    logging.info(
        f"Step 1: Creating all {len(dataset)}×{len(attributes_to_process)} combinations..."
    )

    all_combinations = dataset.map(
        create_all_combinations,
        batched=True,
        load_from_cache_file=True,
        desc="Creating all combinations",
        num_proc=1,
        remove_columns=dataset.column_names,
    )

    total_combinations = len(all_combinations)
    logging.info(f"Created {total_combinations} total combinations")

    # Step 2: Load model
    batch_size = model_config.get("batch_size", 32)  # Larger batch for efficiency
    qwen_processor = QwenPrompting(
        model_path=model_config["model_path"],
        device=device,
        batch_size=batch_size,
        processor=processor,
    )

    logging.info("Step 2: Loading model...")
    qwen_processor.load_model()

    # Step 3: Ultra-fast batch inference
    logging.info("Step 3: Running batch inference on all combinations...")
    all_results = []

    for i in tqdm(range(0, total_combinations, batch_size), desc="Batch inference"):
        batch_end = min(i + batch_size, total_combinations)
        batch = all_combinations[i:batch_end]

        try:
            # Apply chat template to entire batch
            model_inputs = processor.apply_chat_template(
                batch["conversation"],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

            # Batch generate
            with torch.no_grad():
                generated_ids = qwen_processor.model.generate(
                    **model_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                    or processor.tokenizer.eos_token_id,
                )

            # Decode responses
            input_length = model_inputs["input_ids"].shape[1]
            responses = processor.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
            )

            # Store results
            for j, response in enumerate(responses):
                all_results.append(
                    {
                        "image_path": batch["image_path"][j],
                        "attribute": batch["attribute"][j],
                        "response": response.strip(),
                        "label": batch["label"][j],
                    }
                )

        except Exception as e:
            logging.error(f"Error in batch {i}-{batch_end}: {e}")
            # Add empty results for failed batch
            for j in range(len(batch["image_path"])):
                all_results.append(
                    {
                        "image_path": batch["image_path"][j],
                        "attribute": batch["attribute"][j],
                        "response": "",
                        "label": batch["label"][j],
                    }
                )

    results = all_results
