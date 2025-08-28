import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    AutoModelForImageTextToText,
)
from tqdm import tqdm
import logging
from typing import Union, List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from datasets import Dataset, load_from_disk
from pathlib import Path
from PIL import Image
from functools import partial
from operator import attrgetter
import os
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    def __init__(self, device="auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    @abstractmethod
    def extract_features(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from dataset."""
        pass

    def load_layer_features(
        self,
        dataset: Dataset,
        layer: str,
        features_dir: str = "/home/bzq999/data/compmech/features/",
    ):
        if isinstance(layer, list):
            layer = layer[0]

        model_name = getattr(self, "model_name", "unknown_model")
        model_features_dir = Path(features_dir) / model_name
        if not model_features_dir.exists():
            raise FileNotFoundError(
                f"Features directory does not exist: {model_features_dir}"
            )
        feature_dataset_path = model_features_dir / f"layer_{layer}.pt"
        logging.info(f"Loading cached features for layer {layer}")
        cached_layers_features = {}
        try:
            cached_layers_features[layer] = torch.load(
                feature_dataset_path, weights_only=False
            )
            logging.info(
                f"Loaded {len(cached_layers_features[layer])} cached features for layer {layer}"
            )
        except Exception as e:
            logging.warning(f"Failed to load cached features for layer {layer}: {e}")

        merged_dataset = dataset
        features = cached_layers_features[layer]
        feature_column_name = f"layer_{layer}"
        feature_values = []

        for sample in dataset:
            image_path = sample["image_path"]
            if image_path in features:
                feature_values.append(features[image_path].tolist())
            else:
                logging.warning(
                    f"No features found for {image_path}, using zero vector"
                )
                feature_dim = len(list(features.values())[0]) if features else 768
                feature_values.append(np.zeros(feature_dim).tolist())

        merged_dataset = merged_dataset.add_column(feature_column_name, feature_values)

        return merged_dataset

    def extract_and_save(
        self,
        dataset: Dataset,
        features_dir: str = "",
        language_model: str = None,
    ):
        logging.info(
            f"Extracting features from all layers for {len(dataset)} samples..."
        )
        model_name = getattr(self, "model_name", "unknown_model")
        model_features_dir = Path(features_dir) / model_name
        model_features_dir.mkdir(parents=True, exist_ok=True)

        all_layers_features = self.extract_features(dataset)

        # Save all extracted features
        for layer_name, layer_features in all_layers_features.items():
            feature_dataset_path = model_features_dir / f"{layer_name}.pt"
            torch.save(layer_features, feature_dataset_path)
            logging.info(f"Saved features for {layer_name} to {feature_dataset_path}")


class FeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using DINOv2 models."""

    def __init__(
        self,
        model_name=None,
        model_path="facebook/dinov2-base",
        batch_size=16,
        device="auto",
        extract_language=False,
        tower_name=None,
        projection_name=None,
    ):
        """
        Initialize feature extractor.

        Args:
            model_name: HuggingFace model name or path
            batch_size: Batch size for inference
            device: Device to run on
        """
        super().__init__(device)
        self.model_name = model_name or Path(model_path).stem
        self.batch_size = batch_size

        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.vision_tower = tower_name
        self.projection = projection_name

        logging.info(f"Loaded model from {model_path} on {self.device}")
        self.prompt = "USER: <image>\nQuestion: Regarding the main object in the image, is the following statement true or false? The object has the attribute: '{attribute_name}'. Answer with only the word 'True' or 'False'.\nASSISTANT:"

    def _preprocess_image_batch(self, examples):
        """
        Loads images from paths and processes them using the given processor.
        This function is designed to be mapped over a HuggingFace Dataset.
        """
        image_paths = examples["image_path"]

        images = [Image.open(path).convert("RGB") for path in image_paths]

        pixel_values = self.processor(images=images, return_tensors="pt")[
            "pixel_values"
        ]
        return {
            "image_path": image_paths,
            "pixel_values": pixel_values,
        }

    def extract_features(self, dataset) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract features from all layers of the dataset using DINOv2.

        Args:
            dataset: HuggingFace Dataset to extract features from

        Returns:
            Dict mapping layer names to {image_path: features} dictionaries
        """
        logging.info(f"Extracting features from all layers of {self.model_name}...")

        # Dictionary to store all layer embeddings: {layer_name: {image_path: features}}
        all_layers_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

        processed_dataset = dataset.map(
            self._preprocess_image_batch,
            batched=True,
            load_from_cache_file=True,
            desc="Preprocessing Images",
        )
        processed_dataset.set_format(
            type="torch", columns=["pixel_values", "image_path"]
        )

        dataloader = DataLoader(
            processed_dataset, batch_size=self.batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc="Extracting All Layer Features", unit="batch"
            ):
                pixel_values = batch["pixel_values"].to(self.device)
                batch_image_paths: List[str] = [
                    p.item() if torch.is_tensor(p) else p for p in batch["image_path"]
                ]
                if self.vision_tower:
                    model = getattr(self.model, self.vision_tower)
                    outputs = model(pixel_values, output_hidden_states=True)
                else:
                    outputs = self.model(pixel_values, output_hidden_states=True)

                # Extract features from all hidden states
                hidden_states = outputs.hidden_states  # List of tensors for each layer

                for layer_idx, layer_hidden_state in enumerate(hidden_states):
                    layer_name = f"layer_{layer_idx}"
                    cls_token_features = layer_hidden_state[:, 0, :].cpu().numpy()

                    # Initialize layer dict if not exists
                    if layer_name not in all_layers_embeddings:
                        all_layers_embeddings[layer_name] = {}

                    # Store features for each image path
                    for i, path in enumerate(batch_image_paths):
                        all_layers_embeddings[layer_name][path] = cls_token_features[i]

                last_hidden_state = outputs.last_hidden_state
                if not torch.equal(last_hidden_state, outputs["hidden_states"][-1]):
                    last_layer_name = "layer_last"
                    cls_token_features = last_hidden_state[:, 0, :].cpu().numpy()

                    if last_layer_name not in all_layers_embeddings:
                        all_layers_embeddings[last_layer_name] = {}

                    for i, path in enumerate(batch_image_paths):
                        all_layers_embeddings[last_layer_name][path] = (
                            cls_token_features[i]
                        )

                if self.projection:
                    projection = getattr(self.model, self.projection)
                    cls_token_features = projection(outputs.pooler_output).cpu().numpy()

                    if "projection" not in all_layers_embeddings:
                        all_layers_embeddings["projection"] = {}

                    for i, path in enumerate(batch_image_paths):
                        all_layers_embeddings["projection"][path] = cls_token_features[
                            i
                        ]

        return all_layers_embeddings


class LlavaFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using Llava1.5"""

    def __init__(
        self,
        model_name=None,
        model_path="llava-hf/llava-1.5-7b-hf",
        batch_size=32,
        device="auto",
        extract_language=False,
        tower_name=None,
        projection_name=None,
    ):
        """
        Initialize feature extractor.

        Args:
            model_name: HuggingFace model name or path
            batch_size: Batch size for inference
            device: Device to run on
        """
        super().__init__(device)
        self.model_name = model_name or Path(model_path).stem
        self.batch_size = batch_size

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        model = AutoModelForImageTextToText.from_pretrained(model_path)
        self.vision_tower = tower_name
        self.projection = projection_name

        if self.vision_tower:
            self.model = attrgetter(self.vision_tower)(model).to(self.device)
        else:
            self.model = model.to(self.device)
        if self.projection:
            self.projection = attrgetter(self.projection)(model).to(self.device)
        del model

        if extract_language:
            self.extract_features = self.extract_language_features

        logging.info(f"Loaded model from {model_path} on {self.device}")

    def _preprocess_mm_batch(
        self,
        examples,
    ):
        """
        Loads images from paths and processes them using the given processor.
        This function is designed to be mapped over a HuggingFace Dataset.
        """
        image_paths = examples["image_path"]

        images = [Image.open(path).convert("RGB") for path in image_paths]
        text = ["<image>"] * len(images)
        inputs = self.processor(images=images, text=text, return_tensors="pt")
        result = {"image_path": image_paths}
        for key, value in inputs.items():
            result[key] = value
        return result

    def extract_features(self, processed_dataset) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract features from all layers of the dataset using DINOv2.

        Args:
            dataset: HuggingFace Dataset to extract features from

        Returns:
            Dict mapping layer names to {image_path: features} dictionaries
        """
        logging.info(f"Extracting features from all layers of {self.model_name}...")

        # Dictionary to store all layer embeddings: {layer_name: {image_path: features}}
        all_layers_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

        processed_dataset.set_format(
            type="torch", columns=["pixel_values", "image_path"]
        )

        dataloader = DataLoader(
            processed_dataset, batch_size=self.batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc="Extracting All Layer Features", unit="batch"
            ):
                pixel_values = batch["pixel_values"].to(self.device)
                batch_image_paths: List[str] = [
                    p.item() if torch.is_tensor(p) else p for p in batch["image_path"]
                ]
                outputs = self.model(pixel_values, output_hidden_states=True)

                # Extract features from all hidden states
                hidden_states = outputs.hidden_states  # List of tensors for each layer

                for layer_idx, layer_hidden_state in enumerate(hidden_states):
                    layer_name = f"layer_{layer_idx}"

                    if layer_name not in all_layers_embeddings:
                        all_layers_embeddings[layer_name] = {}

                    # Apply global average pooling
                    patch_features = layer_hidden_state[:, 1:, :]  # Remove CLS token
                    pooled_features = (
                        patch_features.mean(dim=1).cpu().numpy()
                    )  # Global average pooling

                    # Store features for each image path
                    for i, path in enumerate(batch_image_paths):
                        all_layers_embeddings[layer_name][path] = pooled_features[i]

                last_hidden_state = outputs.last_hidden_state
                if not torch.equal(last_hidden_state, outputs["hidden_states"][-1]):
                    last_layer_name = "layer_last"
                    # Remove CLS token and apply global average pooling
                    patch_features = last_hidden_state[:, 1:, :]
                    pooled_features = patch_features.mean(dim=1).cpu().numpy()

                    if last_layer_name not in all_layers_embeddings:
                        all_layers_embeddings[last_layer_name] = {}

                    for i, path in enumerate(batch_image_paths):
                        all_layers_embeddings[last_layer_name][path] = pooled_features[
                            i
                        ]

                if self.projection:
                    projected_features = self.projection(hidden_states[-2][:, 1:, :])
                    pooled_features = projected_features.mean(dim=1).cpu().numpy()

                    if "projection" not in all_layers_embeddings:
                        all_layers_embeddings["projection"] = {}

                    for i, path in enumerate(batch_image_paths):
                        all_layers_embeddings["projection"][path] = pooled_features[i]

        return all_layers_embeddings

    def extract_language_features(self, processed_dataset):
        logging.info(f"Extracting features from all layers of {self.model_name}...")

        # Dictionary to store all layer embeddings: {layer_name: {image_path: features}}
        all_layers_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

        processed_dataset.set_format(
            type="torch",
            columns=["pixel_values", "image_path", "input_ids", "attention_mask"],
        )

        dataloader = DataLoader(
            processed_dataset, batch_size=self.batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc="Extracting Language Features", unit="batch"
            ):
                # Prepare inputs for the model
                inputs = {
                    "pixel_values": batch["pixel_values"].to(self.device),
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }
                batch_image_paths: List[str] = [
                    p.item() if torch.is_tensor(p) else p for p in batch["image_path"]
                ]

                # Get language model outputs
                outputs = self.model(**inputs, output_hidden_states=True)
                # Extract features from all hidden states of the language model
                hidden_states = outputs.hidden_states  # List of tensors for each layer
                for layer_idx, layer_hidden_state in enumerate(hidden_states):
                    layer_name = f"lang_layer_{layer_idx}"
                    if layer_name not in all_layers_embeddings:
                        all_layers_embeddings[layer_name] = {}
                    sequence_features = layer_hidden_state.mean(dim=1).cpu().numpy()
                    # Store features for each image path
                    for i, path in enumerate(batch_image_paths):
                        all_layers_embeddings[layer_name][path] = sequence_features[i]

        return all_layers_embeddings


class QwenFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for Qwen2.5-VL using vision encoder with monkey patch."""

    def __init__(
        self,
        model_name=None,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        batch_size=32,
        device="auto",
        extract_language=False,
        tower_name=None,
        projection_name=None,
    ):
        """
        Initialize Qwen2.5-VL feature extractor.

        Args:
            model_name: Name for the model (for saving features)
            model_path: HuggingFace model path
            batch_size: Batch size for inference
            device: Device to run on
        """
        super().__init__(device)
        self.model_name = model_name or Path(model_path).stem
        self.batch_size = batch_size

        # Load processor and model
        from src.models.qwen_vision_patch import patch_qwen_vision_model

        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        model = AutoModelForImageTextToText.from_pretrained(model_path)

        # Apply monkey patch for output_hidden_states support
        model = patch_qwen_vision_model(model)
        self.model = model.model.visual.to(self.device)
        self.model.eval()

        logging.info(f"Loaded Qwen2.5-VL model from {model_path} on {self.device}")

    def extract_features(self, dataset) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract features from all vision layers of Qwen2.5-VL using on-the-fly processing.
        """

        logging.info(f"Extracting features from all layers of {self.model_name}...")

        # Dictionary to store all layer embeddings: {layer_name: {image_path: features}}
        all_layers_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

        # Process dataset in batches on the fly
        num_samples = len(dataset)
        for i in tqdm(
            range(0, num_samples, self.batch_size),
            desc="Extracting Qwen Vision Features",
            unit="batch",
        ):
            batch_end = min(i + self.batch_size, num_samples)
            batch_data = dataset.select(range(i, batch_end))

            # Get image paths and load images
            image_paths = [sample["image_path"] for sample in batch_data]
            images = [Image.open(path).convert("RGB") for path in image_paths]

            # Create messages for batch processing
            messages = []
            for image in images:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": " "},
                        ],
                    }
                ]
                messages.append(message)

            # Preparation for batch inference
            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                # Extract features using the patched vision model
                pixel_values = inputs["pixel_values"].type(self.model.dtype)
                image_grid_thw = inputs["image_grid_thw"]

                final_hidden_states, all_hidden_states = self.model(
                    pixel_values.to(self.device),
                    grid_thw=image_grid_thw.to(self.device),
                    output_hidden_states=True,
                )

                # Process each layer's hidden states
                for layer_idx, layer_hidden_state in enumerate(all_hidden_states):
                    layer_name = f"layer_{layer_idx}"

                    if layer_name not in all_layers_embeddings:
                        all_layers_embeddings[layer_name] = {}

                    # Global average pooling across spatial dimensions
                    # Qwen vision features are [batch_size, num_patches, hidden_size]
                    pooled_features = layer_hidden_state.mean(dim=1).cpu().numpy()

                    # Store features for each image path
                    for j, path in enumerate(image_paths):
                        all_layers_embeddings[layer_name][path] = pooled_features[j]

        return all_layers_embeddings


class VLLMQwenFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for Qwen2.5-VL using VLLM for high-throughput batch processing."""

    def __init__(
        self,
        model_name=None,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        batch_size=32,
        device="auto",
        extract_language=False,
        tower_name=None,
        projection_name=None,
    ):
        """
        Initialize VLLM-based Qwen2.5-VL feature extractor.

        Args:
            model_name: Name for the model (for saving features)
            model_path: HuggingFace model path
            batch_size: Batch size for VLLM inference
            device: Device to run on (handled by VLLM)
        """
        super().__init__(device)
        self.model_name = model_name or Path(model_path).stem
        self.batch_size = batch_size
        self.model_path = model_path

        # Get model-specific VLLM configuration
        self.vllm_config = self._get_model_config(model_path)

        # Initialize VLLM model
        logging.info(f"Loading VLLM model: {model_path}")
        logging.info(f"VLLM config: {self.vllm_config}")
        self.llm = LLM(model=model_path, **self.vllm_config)

        # Access the underlying vision model for feature extraction
        self.vision_model = self._get_vision_model()

        logging.info(f"Loaded VLLM Qwen2.5-VL model from {model_path}")

    def _get_model_config(self, model_path: str) -> dict:
        """Get model-specific VLLM configuration."""
        model_name_lower = model_path.lower()

        if "qwen2.5-vl" in model_name_lower or "qwen2_5" in model_name_lower:
            return {
                "max_model_len": 4096,
                "max_num_seqs": max(
                    1, self.batch_size // 4
                ),  # Adjust based on batch size
                "mm_processor_kwargs": {
                    "min_pixels": 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
                "limit_mm_per_prompt": {"image": 1},
                "trust_remote_code": True,
            }
        else:
            return {
                "max_model_len": 4096,
                "limit_mm_per_prompt": {"image": 1},
                "trust_remote_code": True,
            }

    def _get_vision_model(self):
        """Access the vision model from VLLM's internal structure."""
        try:
            # Try different VLLM internal structures based on version
            llm_engine = self.llm.llm_engine
            
            # Try newer VLLM structure first
            if hasattr(llm_engine, 'engine'):
                engine = llm_engine.engine
                if hasattr(engine, 'model_executor'):
                    model_executor = engine.model_executor
                elif hasattr(engine, 'worker'):
                    # Single worker case
                    worker = engine.worker
                    model = worker.model
                else:
                    raise AttributeError("Could not find model executor in engine")
            elif hasattr(llm_engine, 'model_executor'):
                # Older VLLM structure
                model_executor = llm_engine.model_executor
            else:
                # Try direct access to worker
                if hasattr(llm_engine, 'worker'):
                    worker = llm_engine.worker
                    model = worker.model
                else:
                    raise AttributeError("Could not find model executor or worker")
            
            # Get worker and model if we have model_executor
            if 'model_executor' in locals():
                if hasattr(model_executor, "driver_worker"):
                    # Single GPU case
                    worker = model_executor.driver_worker
                elif hasattr(model_executor, "workers"):
                    # Multi GPU case - get first worker
                    worker = list(model_executor.workers.values())[0]
                else:
                    raise AttributeError("Could not find worker in model executor")
                
                # Get the actual model
                model = worker.model_runner.model if hasattr(worker, 'model_runner') else worker.model

            # Access the vision encoder
            if hasattr(model, "model") and hasattr(model.model, "visual"):
                vision_model = model.model.visual
                vision_model.eval()
                return vision_model
            elif hasattr(model, "visual"):
                vision_model = model.visual
                vision_model.eval()
                return vision_model
            else:
                # Log model structure for debugging
                logging.error(f"Model structure: {type(model)}")
                if hasattr(model, '__dict__'):
                    logging.error(f"Model attributes: {list(model.__dict__.keys())}")
                raise AttributeError("Could not find vision model in VLLM structure")

        except Exception as e:
            logging.error(f"Failed to access vision model from VLLM: {e}")
            # Try to provide more debugging info
            logging.error(f"LLM engine type: {type(self.llm.llm_engine)}")
            if hasattr(self.llm.llm_engine, '__dict__'):
                logging.error(f"LLM engine attributes: {list(self.llm.llm_engine.__dict__.keys())}")
            raise

    def _create_vllm_prompts(self, image_paths: list) -> list:
        """Create VLLM-compatible prompts for feature extraction."""
        # Simple prompt for feature extraction (we just need to process the images)
        prompt_text = (
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "<|im_end|>\n"
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

    def extract_features(self, dataset) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract features from all vision layers using VLLM for efficient processing.
        """
        logging.info(
            f"Extracting features from all layers of {self.model_name} using VLLM..."
        )

        # Dictionary to store all layer embeddings: {layer_name: {image_path: features}}
        all_layers_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

        # Create sampling params (we don't actually need the generated text)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,  # Minimal tokens since we only need features
            stop_token_ids=None,
        )

        # Process dataset in batches using VLLM
        num_samples = len(dataset)
        for i in tqdm(
            range(0, num_samples, self.batch_size),
            desc="Extracting VLLM Qwen Vision Features",
            unit="batch",
        ):
            batch_end = min(i + self.batch_size, num_samples)
            batch_data = dataset.select(range(i, batch_end))

            # Get image paths
            image_paths = [sample["image_path"] for sample in batch_data]

            # Create VLLM prompts
            vllm_prompts = self._create_vllm_prompts(image_paths)

            try:
                with torch.no_grad():
                    # Use VLLM to process the batch (this handles all preprocessing efficiently)
                    # We'll intercept at the vision model level to get features
                    self._extract_batch_features(
                        vllm_prompts, image_paths, all_layers_embeddings
                    )

            except Exception as e:
                logging.error(f"Error processing batch {i}-{batch_end}: {e}")
                # Continue with next batch
                continue

        return all_layers_embeddings

    def _extract_batch_features(
        self, vllm_prompts: list, image_paths: list, all_layers_embeddings: dict
    ):
        """Extract features from a batch using VLLM's internal processing."""

        try:
            # Extract images from prompts
            images = [prompt["multi_modal_data"]["image"] for prompt in vllm_prompts]
            texts = [prompt["prompt"] for prompt in vllm_prompts]

            # Try to get processor from VLLM's internal structure
            processor = None
            try:
                # Try different ways to access the processor
                llm_engine = self.llm.llm_engine
                
                if hasattr(llm_engine, 'engine'):
                    engine = llm_engine.engine
                    if hasattr(engine, 'worker'):
                        worker = engine.worker
                        if hasattr(worker, 'model') and hasattr(worker.model, 'processor'):
                            processor = worker.model.processor
                elif hasattr(llm_engine, 'worker'):
                    worker = llm_engine.worker
                    if hasattr(worker, 'model') and hasattr(worker.model, 'processor'):
                        processor = worker.model.processor
                elif hasattr(llm_engine, 'model_executor'):
                    model_executor = llm_engine.model_executor
                    if hasattr(model_executor, "driver_worker"):
                        worker = model_executor.driver_worker
                    else:
                        worker = list(model_executor.workers.values())[0]
                    
                    if hasattr(worker, 'model_runner') and hasattr(worker.model_runner, 'model'):
                        if hasattr(worker.model_runner.model, 'processor'):
                            processor = worker.model_runner.model.processor
                
                if processor is None:
                    # Fallback: create processor directly
                    from transformers import AutoProcessor
                    processor = AutoProcessor.from_pretrained(self.model_path)
                    
            except Exception as e:
                logging.warning(f"Could not access VLLM processor, using fallback: {e}")
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(self.model_path)

            # Process through the processor
            inputs = processor(
                text=texts, images=images, return_tensors="pt", padding=True
            )

            # Move to appropriate device
            device = next(self.vision_model.parameters()).device
            pixel_values = inputs["pixel_values"].to(device)
            image_grid_thw = inputs["image_grid_thw"].to(device)

            # Extract features using the vision model
            pixel_values = pixel_values.type(self.vision_model.dtype)
            final_hidden_states, all_hidden_states = self.vision_model(
                pixel_values, grid_thw=image_grid_thw, output_hidden_states=True
            )

            # Process each layer's hidden states
            for layer_idx, layer_hidden_state in enumerate(all_hidden_states):
                layer_name = f"layer_{layer_idx}"

                if layer_name not in all_layers_embeddings:
                    all_layers_embeddings[layer_name] = {}

                # Global average pooling across spatial dimensions
                pooled_features = layer_hidden_state.mean(dim=1).cpu().numpy()

                # Store features for each image path
                for j, path in enumerate(image_paths):
                    all_layers_embeddings[layer_name][path] = pooled_features[j]

        except Exception as e:
            logging.error(f"Error in _extract_batch_features: {e}")
            raise


def get_feature_extractor(extractor_type: str, **kwargs) -> BaseFeatureExtractor:
    """Factory function to create feature extractors."""
    extractors = {
        "dinov2": FeatureExtractor,
        "clip": FeatureExtractor,
        "llava": LlavaFeatureExtractor,
        # "qwen2.5-vl": QwenFeatureExtractor,
        "qwen2.5-vl": VLLMQwenFeatureExtractor,
    }

    if extractor_type not in extractors:
        raise ValueError(
            f"Unknown extractor type: {extractor_type}. Available: {list(extractors.keys())}"
        )

    return extractors[extractor_type](**kwargs)
