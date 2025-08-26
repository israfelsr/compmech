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
        features_dir: str = "/home/bzq999/data/compmech/features/",
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
        from transformers import Qwen2VLForConditionalGeneration
        from src.models.qwen_vision_patch import patch_qwen_vision_model
        
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path)
        
        # Apply monkey patch for output_hidden_states support
        self.model = patch_qwen_vision_model(model).to(self.device)
        self.model.eval()

        logging.info(f"Loaded Qwen2.5-VL model from {model_path} on {self.device}")

    def _preprocess_qwen_batch(self, examples):
        """
        Preprocess images for Qwen2.5-VL using conversation format.
        """
        image_paths = examples["image_path"]
        images = [Image.open(path).convert("RGB") for path in image_paths]
        
        # Create conversations for each image
        conversations = []
        for image in images:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image."},  # Dummy text
                    ],
                }
            ]
            conversations.append(conversation)
        
        # Process all conversations
        processed_inputs = []
        for conversation in conversations:
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                padding=True,
                return_tensors="pt",
            )
            processed_inputs.append(inputs)
        
        # Batch the inputs
        result = {"image_path": image_paths}
        
        # Stack pixel values and image_grid_thw
        if processed_inputs:
            result["pixel_values"] = torch.cat([inp.pixel_values for inp in processed_inputs], dim=0)
            result["image_grid_thw"] = torch.cat([inp.image_grid_thw for inp in processed_inputs], dim=0)
        
        return result

    def extract_features(self, dataset) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract features from all vision layers of Qwen2.5-VL.
        """
        logging.info(f"Extracting features from all layers of {self.model_name}...")

        # Dictionary to store all layer embeddings: {layer_name: {image_path: features}}
        all_layers_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

        processed_dataset = dataset.map(
            self._preprocess_qwen_batch,
            batched=True,
            load_from_cache_file=True,
            desc="Preprocessing Images for Qwen",
        )
        processed_dataset.set_format(
            type="torch", columns=["pixel_values", "image_grid_thw", "image_path"]
        )

        dataloader = DataLoader(
            processed_dataset, batch_size=self.batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc="Extracting Qwen Vision Features", unit="batch"
            ):
                pixel_values = batch["pixel_values"].to(self.device)
                image_grid_thw = batch["image_grid_thw"].to(self.device)
                batch_image_paths: List[str] = [
                    p.item() if torch.is_tensor(p) else p for p in batch["image_path"]
                ]

                # Extract features using the patched vision model
                pixel_values = pixel_values.type(self.model.model.visual.dtype)
                final_hidden_states, all_hidden_states = self.model.model.visual(
                    pixel_values, grid_thw=image_grid_thw, output_hidden_states=True
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
                    for i, path in enumerate(batch_image_paths):
                        all_layers_embeddings[layer_name][path] = pooled_features[i]

        return all_layers_embeddings


def get_feature_extractor(extractor_type: str, **kwargs) -> BaseFeatureExtractor:
    """Factory function to create feature extractors."""
    extractors = {
        "dinov2": FeatureExtractor,
        "clip": FeatureExtractor,
        "llava": LlavaFeatureExtractor,
        "qwen": QwenFeatureExtractor,
    }

    if extractor_type not in extractors:
        raise ValueError(
            f"Unknown extractor type: {extractor_type}. Available: {list(extractors.keys())}"
        )

    return extractors[extractor_type](**kwargs)
