import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import logging
from typing import Union, List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from datasets import Dataset, load_from_disk
from pathlib import Path
from PIL import Image
from functools import partial
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

    @staticmethod
    def _preprocess_image_batch(examples, processor):
        """
        Loads images from paths and processes them using the given processor.
        This function is designed to be mapped over a HuggingFace Dataset.
        """
        image_paths = examples["image_path"]

        images = [Image.open(path).convert("RGB") for path in image_paths]

        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        return {
            "image_path": image_paths,
            "pixel_values": pixel_values,
        }

    def load_layer_features(self, dataset, layer, features_dir):
        if isinstance(layer, list):
            layer = layer[0]

        model_name = getattr(self, "model_name", "unknown_model")
        model_features_dir = Path(features_dir) / model_name
        if not model_features_dir.exists():
            raise FileNotFoundError(f"Features directory does not exist: {model_features_dir}")
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

    def add_features_to_dataset(
        self,
        dataset: Dataset,
        layers: Union[str, int, List[Union[str, int]]] = "last",
        features_dir: str = "/home/bzq999/data/compmech/features/",
    ) -> Dataset:
        """
        Add feature columns to a HuggingFace dataset.

        Args:
            dataset: Input dataset with 'image_path' column
            layers: Layer(s) to extract features from
            features_dir: Base directory to save feature datasets

        Returns:
            Dataset with added feature columns
        """
        # Ensure layers is a list
        if not isinstance(layers, list):
            layers = [layers]

        logging.info(
            f"Extracting features from layers {layers} for {len(dataset)} samples..."
        )

        model_name = getattr(self, "model_name", "unknown_model")
        model_features_dir = Path(features_dir) / model_name
        model_features_dir.mkdir(parents=True, exist_ok=True)

        # Check if we need to extract any features
        layers_to_extract = []
        cached_layers_features = {}

        for layer in layers:
            feature_dataset_path = model_features_dir / f"layer_{layer}.pt"
            if feature_dataset_path.exists():
                logging.info(f"Loading cached features for layer {layer}")
                try:
                    cached_layers_features[layer] = torch.load(
                        feature_dataset_path, weights_only=False
                    )
                    logging.info(
                        f"Loaded {len(cached_layers_features[layer])} cached features for layer {layer}"
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to load cached features for layer {layer}: {e}"
                    )
                    layers_to_extract.append(layer)
            else:
                layers_to_extract.append(layer)

        # Extract all features at once if needed
        if layers_to_extract:
            logging.info(f"Extracting features for {len(layers_to_extract)} layers...")
            all_layers_features = self.extract_features(dataset)

            # Save all extracted features
            for layer_name, layer_features in all_layers_features.items():
                feature_dataset_path = model_features_dir / f"{layer_name}.pt"
                torch.save(layer_features, feature_dataset_path)
                logging.info(
                    f"Saved features for {layer_name} to {feature_dataset_path}"
                )

                # Add to cached features if it's one of our requested layers
                layer_key = (
                    layer_name
                    if layer_name == "last"
                    else int(layer_name.split("_")[1])
                )
                if layer_key in layers:
                    cached_layers_features[layer_key] = layer_features

        # Add feature columns to dataset
        merged_dataset = dataset
        for layer in layers:
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

            merged_dataset = merged_dataset.add_column(
                feature_column_name, feature_values
            )

        return merged_dataset


class FeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using DINOv2 models."""

    def __init__(
        self,
        model_name=None,
        model_path="facebook/dinov2-base",
        batch_size=32,
        device="auto",
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

        bound_preprocess_function = partial(
            self._preprocess_image_batch, processor=self.processor
        )

        processed_dataset = dataset.map(
            bound_preprocess_function,
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
                    norm_tensor = last_hidden_state.norm(p=2, dim=-1, keepdim=True)
                    last_hidden_state = last_hidden_state / norm_tensor
                    cls_token_features = (
                        projection(last_hidden_state)[:, 0, :].cpu().numpy()
                    )

                    if "projection" not in all_layers_embeddings:
                        all_layers_embeddings["projection"] = {}

                    for i, path in enumerate(batch_image_paths):
                        all_layers_embeddings["projection"][path] = cls_token_features[
                            i
                        ]

        return all_layers_embeddings


class CLIPFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using CLIP models."""

    def __init__(
        self, model_name="openai/clip-vit-base-patch32", batch_size=32, device="auto"
    ):
        super().__init__(device)
        self.model_name = model_name
        self.batch_size = batch_size

        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logging.info(f"Loaded {model_name} on {self.device}")

    def extract_features(self, dataset, layer) -> Tuple[np.ndarray, np.ndarray]:
        """Extract visual features using CLIP."""
        logging.info(f"Extracting CLIP visual features...")

        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting CLIP features"):
                images = images.to(self.device)

                # Get visual features
                vision_outputs = self.model.vision_model(pixel_values=images)
                features = vision_outputs.pooler_output

                all_features.append(features.cpu())
                all_labels.append(labels.cpu())

        features = torch.cat(all_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

        logging.info(f"Extracted CLIP features shape: {features.shape}")

        return features, labels


class MultiLayerFeatureExtractor(BaseFeatureExtractor):
    """Extract features from multiple layers and concatenate them."""

    def __init__(self, base_extractor: BaseFeatureExtractor, layers: list):
        super().__init__(base_extractor.device)
        self.base_extractor = base_extractor
        self.layers = layers

    def extract_features(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and concatenate features from multiple layers."""
        all_layer_features = []
        labels = None

        for layer in self.layers:
            self.base_extractor.layer = layer
            features, labels = self.base_extractor.extract_features(dataset)
            all_layer_features.append(features)

        # Concatenate features from all layers
        combined_features = np.concatenate(all_layer_features, axis=1)

        logging.info(
            f"Combined features from {len(self.layers)} layers: {combined_features.shape}"
        )

        return combined_features, labels


def get_feature_extractor(extractor_type: str, **kwargs) -> BaseFeatureExtractor:
    """Factory function to create feature extractors."""
    extractors = {
        "dinov2": FeatureExtractor,
        "swinv2": FeatureExtractor,
        "clip": FeatureExtractor,
    }

    if extractor_type not in extractors:
        raise ValueError(
            f"Unknown extractor type: {extractor_type}. Available: {list(extractors.keys())}"
        )

    return extractors[extractor_type](**kwargs)
