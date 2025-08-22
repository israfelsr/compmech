"""
Model utilities for VLM prompting.
Supports both VLLM and HuggingFace models for attribute testing.
"""

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    GenerationConfig,
)

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

from PIL import Image
import logging


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


def query_model(
    model_name: str,
    model,
    processor,
    prompt,
    images,
    device: str = "cuda",
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
):
    """
    Query the model based on the model name and type.
    """
    if model_name in VLLM_MODELS:
        return query_vllm(model, processor, prompt, images, max_tokens)
    elif model_name in HUGGINGFACE_MODELS:
        return query_huggingface(model, processor, prompt, images, device, max_tokens)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def query_vllm(model, processor, prompt, images, max_tokens=MAX_TOKENS):
    """Query VLLM models."""
    sampling_params = SamplingParams(
        max_tokens=max_tokens, temperature=TEMPERATURE, top_p=0.9, stop_token_ids=None
    )

    # Prepare inputs
    if images is not None and len(images) > 0:
        try:
            # Load and resize images
            pil_images = []
            for img_path in images:
                img = Image.open(img_path).convert("RGB")
                # Resize to reasonable size for efficiency
                img = img.resize((512, 512))
                pil_images.append(img)

            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": pil_images},
            }
        except Exception as e:
            logging.error(f"Error loading images {images}: {e}")
            inputs = {"prompt": prompt}
    else:
        inputs = {"prompt": prompt}

    # Generate response
    try:
        with torch.inference_mode():
            outputs = model.generate([inputs], sampling_params)
            response = outputs[0].outputs[0].text.strip()
        return response, None
    except Exception as e:
        logging.error(f"VLLM generation error: {e}")
        return f"ERROR: {str(e)}", None


def query_huggingface(
    model, processor, prompt, images, device="cuda", max_tokens=MAX_TOKENS
):
    """Query HuggingFace transformers models."""
    try:
        # Load image if provided
        if images is not None and len(images) > 0:
            image = Image.open(images[0]).convert("RGB")
        else:
            image = None

        # Process inputs
        if isinstance(prompt, list):
            # Convert chat format to string if needed
            if len(prompt) == 1 and "content" in prompt[0]:
                prompt_text = prompt[0]["content"]
            else:
                prompt_text = str(prompt)
        else:
            prompt_text = prompt

        inputs = processor(
            images=image, text=prompt_text, return_tensors="pt", padding=True
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=TEMPERATURE,
                do_sample=False,  # Deterministic for attribute testing
                pad_token_id=(
                    processor.tokenizer.eos_token_id
                    if processor.tokenizer.eos_token_id is not None
                    else processor.tokenizer.pad_token_id
                ),
            )

            # Decode only new tokens
            response = processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )[0].strip()

        return response, None

    except Exception as e:
        logging.error(f"HuggingFace generation error: {e}")
        return f"ERROR: {str(e)}", None


def format_prompt_for_model(model_name: str, base_prompt: str):
    """Format prompt according to model's expected format."""
    if model_name in ["qwen2.5-7b", "qwen2.5-3b", "qwen2.5-32b", "qwen2.5-72b"]:
        # Qwen chat format
        return [{"role": "user", "content": base_prompt}]
    elif model_name == "molmo":
        # Molmo uses simple text format
        return base_prompt
    elif model_name == "llava" or "llava" in model_name.lower():
        # LLaVA uses text format
        return base_prompt
    else:
        return base_prompt


def get_model_type(model_name: str) -> str:
    """Get model type (vllm or huggingface)."""
    if model_name in VLLM_MODELS:
        return "vllm"
    elif model_name in HUGGINGFACE_MODELS:
        return "huggingface"
    else:
        raise ValueError(f"Unknown model type for {model_name}")
