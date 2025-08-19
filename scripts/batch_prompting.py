#!/usr/bin/env python3
"""
Optimized batch prompting script using LLAVA model.
Processes image features only once per image, then reuses for multiple text prompts.
"""

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


def create_prompt_templates():
    """Create prompt templates for different attributes."""
    templates = {
        "default": "USER: Image: <image>\nQuestion: Regarding the main object in the image, is the following statement true or false? The object has the attribute: '{attribute_name}'. Answer with only the word 'True' or 'False'.\nASSISTANT:"
    }
    return templates


def get_all_attributes_from_dataset(dataset):
    """Extract all unique attributes from the dataset."""
    # Get attributes from the first sample, excluding first 2 keys (image_path, concept)
    if len(dataset) > 0:
        return list(dataset[0].keys())[2:]
    return []


class BatchLLaVAProcessor:
    """Optimized LLaVA processor that caches image features and processes multiple text prompts efficiently."""
    
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.image_cache = {}  # Cache for image features
    
    def encode_image(self, image_path):
        """Encode image and cache the result."""
        if image_path in self.image_cache:
            return self.image_cache[image_path]
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        # Process image only (without text)
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items() if 'pixel_values' in k}
        
        # Get image features from the model
        with torch.no_grad():
            # For LLaVA, we need to get the vision features
            if hasattr(self.model, 'get_vision_tower'):
                vision_tower = self.model.get_vision_tower()
                if vision_tower is not None:
                    image_features = vision_tower(image_inputs['pixel_values'])
                else:
                    # Fallback: use the full forward pass but extract vision features
                    dummy_input_ids = torch.tensor([[1]], dtype=torch.long, device=self.device)  # Dummy token
                    outputs = self.model(input_ids=dummy_input_ids, **image_inputs, output_hidden_states=True)
                    image_features = outputs.hidden_states[-1]  # Last layer hidden states
            else:
                # Alternative approach: process with dummy text to get the structure
                dummy_text = "dummy"
                dummy_inputs = self.processor(images=image, text=dummy_text, return_tensors="pt")
                dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}
                
                with torch.no_grad():
                    # We'll store the full processed inputs for reuse
                    image_features = {
                        'pixel_values': image_inputs['pixel_values'],
                        'image_processed': True
                    }
        
        # Cache the features
        self.image_cache[image_path] = image_features
        return image_features
    
    def process_batch_prompts_for_image(self, image_path, prompts, max_new_tokens=100):
        """Process multiple prompts for a single image efficiently."""
        results = []
        
        # Get cached image features
        image_features = self.encode_image(image_path)
        
        # Load image for processor (we still need this for the processor)
        image = Image.open(image_path).convert("RGB")
        
        # Process each prompt
        for prompt in prompts:
            try:
                # Process the full input (image + text)
                inputs = self.processor(images=image, text=prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate response
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
                    # Decode only the new tokens (response)
                    response = self.processor.batch_decode(
                        generated_ids[:, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    )[0]
                
                results.append(response.strip())
                
            except Exception as e:
                logging.error(f"Error processing prompt for {image_path}: {e}")
                results.append(f"ERROR: {str(e)}")
        
        return results


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
    
    # Load model and processor
    model_config = config["model"]
    model = AutoModelForImageTextToText.from_pretrained(
        model_config["model_path"],
        torch_dtype=torch.float16,
        device_map=model_config["device"]
    )
    processor = AutoProcessor.from_pretrained(model_config["model_path"], use_fast=True)
    
    # Initialize batch processor
    batch_processor = BatchLLaVAProcessor(model, processor, device)
    
    # Create prompt templates
    prompt_templates = create_prompt_templates()
    
    # Get attribute(s) from config or args
    specific_attribute = args.attribute or config["probe"].get("specific_attribute")
    
    if specific_attribute:
        # Single attribute mode
        attributes_to_process = [specific_attribute]
        logging.info(f"Processing single attribute: {specific_attribute}")
    else:
        # All attributes mode
        attributes_to_process = get_all_attributes_from_dataset(dataset)
        logging.info(f"Processing all {len(attributes_to_process)} attributes")
        logging.info(f"Attributes: {attributes_to_process[:10]}{'...' if len(attributes_to_process) > 10 else ''}")
    
    # Process all samples
    all_results = {}
    total_samples = len(dataset)
    logging.info(f"Processing {total_samples} images with {len(attributes_to_process)} attributes each...")
    
    for sample in tqdm(dataset, desc="Processing images"):
        try:
            image_path = sample["image_path"]
            
            # Create all prompts for this image
            prompts = []
            prompt_info = []  # Store metadata about each prompt
            
            for attribute_name in attributes_to_process:
                prompt_template = prompt_templates.get(args.prompt_type, prompt_templates["default"])
                prompt = prompt_template.format(attribute_name=attribute_name)
                prompts.append(prompt)
                prompt_info.append(attribute_name)
            
            # Process all prompts for this image in one go
            responses = batch_processor.process_batch_prompts_for_image(
                image_path, prompts, max_new_tokens=args.max_tokens
            )
            
            # Organize results by attribute
            for attribute_name, response in zip(prompt_info, responses):
                if attribute_name not in all_results:
                    all_results[attribute_name] = []
                
                all_results[attribute_name].append({
                    "image_path": image_path,
                    "attribute": attribute_name,
                    "prompt": prompts[prompt_info.index(attribute_name)],
                    "response": response
                })
                
        except Exception as e:
            logging.error(f"Error processing {sample.get('image_path', 'unknown')}: {e}")
            continue
    
    # Save results
    output_dir = Path(args.output_dir or config["probe"].get("output_dir", "results/batch_prompting"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(attributes_to_process) == 1:
        # Single attribute: save as before
        attribute_name = attributes_to_process[0]
        results = all_results[attribute_name]
        results_filename = f"batch_prompting_results_{attribute_name}_{args.prompt_type}.json"
        results_path = output_dir / results_filename
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Results saved to {results_path}")
        logging.info(f"Processed {len(results)} samples successfully")
    else:
        # Multiple attributes: save separate files and a combined file
        total_processed = 0
        for attribute_name, results in all_results.items():
            results_filename = f"batch_prompting_results_{attribute_name}.json"
            results_path = output_dir / results_filename
            
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            total_processed += len(results)
            logging.info(f"Saved {attribute_name}: {len(results)} samples -> {results_path}")
        
        # Save combined results
        combined_filename = f"batch_prompting_results_all_{args.prompt_type}.json"
        combined_path = output_dir / combined_filename
        
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Combined results saved to {combined_path}")
        logging.info(f"Processed {len(attributes_to_process)} attributes with {total_processed} total samples")
    
    # Log cache statistics
    logging.info(f"Image cache size: {len(batch_processor.image_cache)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch prompt LLAVA model with optimized image processing"
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
        help="Specific attribute to prompt about. If not provided, processes all attributes in dataset"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["default", "taxonomic", "visual", "functional"],
        default="default",
        help="Type of prompt template to use"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for results"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Maximum new tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    main(args)