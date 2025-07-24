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

    def _preprocess_image_batch(examples, processor):
        """
        Loads images from paths and processes them using the given processor.
        This function is designed to be mapped over a HuggingFace Dataset.
        """
        image_paths = examples["image_path"]
        labels = examples["label"]  # Assuming 'label' column exists

        images = [Image.open(path).convert("RGB") for path in image_paths]

        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        return {
            "image_path": image_paths,
            "pixel_values": pixel_values,
            "labels": labels,
        }

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
            batch_size: Processing batch size
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

        merged_dataset = dataset

        # Process each layer and save
        for layer in layers:
            model_name = getattr(self, "model_name", "unknown_model")
            feature_dataset_path = Path(features_dir) / model_name / f"layer_{layer}.pt"
            if feature_dataset_path.exists():
                logging.info(
                    f"Loading existing feature dataset from {feature_dataset_path}"
                )
                try:
                    features = torch.load(feature_dataset_path)
                    logging.info(f"Loaded {len(features)} feature samples from cache")
                except Exception as e:
                    logging.warning(
                        f"Failed to load cached features: {e}. Recomputing..."
                    )
                    features = self.extract_features(dataset, layer)
                    torch.save(features, feature_dataset_path)
                    logging.info(
                        f"Features computed and saved in {feature_dataset_path}"
                    )

            else:
                logging.info(f"Extracting features for layer '{layer}'...")
                features = self.extract_features(dataset, layer)
                torch.save(features, feature_dataset_path)
                logging.info(f"Features computed and saved in {feature_dataset_path}")

            # Merge with original dataset
            feature_column_name = f"layer_{layer}"
            feature_values = []
            for sample in dataset:
                image_path = sample["image_path"]
                if image_path in features:
                    feature_values.append(features[image_path])
                else:
                    logging.warning(
                        f"No features found for {image_path}, using zero vector"
                    )
                    feature_dim = len(list(features.values())[0]) if features else 768
                    feature_values.append(np.zeros(feature_dim).tolist())

            merged_dataset = dataset.add_column(feature_column_name, feature_values)

        return merged_dataset


class DINOv2FeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using DINOv2 models."""

    def __init__(
        self,
        model_name="None",
        model_path="facebook/dinov2-base",
        batch_size=32,
        device="auto",
    ):
        """
        Initialize DINOv2 feature extractor.

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

        logging.info(f"Loaded model from {model_path} on {self.device}")

    def extract_features(self, dataset, layer) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from dataset using DINOv2.

        Args:
            dataset: HuggingFace Dataset to extract features from

        Returns:
            tuple: (features, labels) as numpy arrays
        """
        logging.info(
            f"Extracting features from {self.model_name} ({self.layer} layer)..."
        )

        all_embeddings_dict: Dict[str, np.ndarray] = {}

        bound_preprocess_function = partial(
            self._preprocess_image_batch, processor=self.processor
        )

        processed_dataset = dataset.map(
            bound_preprocess_function,
            batched=True,
            remove_columns=["image_path"],
            num_proc=max(1, os.cpu_count() - 1),
            load_from_cache_file=True,
            desc="Preprocessing Images",
        )
        processed_dataset.set_format(
            type="torch", columns=["pixel_values", "labels", "image_path"]
        )

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(
                data_loader, desc=f"Extracting Layer {layer} Features", unit="batch"
            ):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"]
                batch_image_paths: List[str] = [
                    p.item() if torch.is_tensor(p) else p for p in batch["image_path"]
                ]
                outputs = self.model(pixel_values, output_hidden_states=True)

                layer_hidden_state = outputs.hidden_states[layer]
                cls_token_features = layer_hidden_state[:, 0, :].cpu().numpy()

                # Append features and labels to the overall lists
                for i, path in enumerate(batch_image_paths):
                    all_embeddings_dict[path] = cls_token_features[i]

        return all_embeddings_dict


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

    def extract_features(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
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
        "dinov2": DINOv2FeatureExtractor,
        "clip": CLIPFeatureExtractor,
    }

    if extractor_type not in extractors:
        raise ValueError(
            f"Unknown extractor type: {extractor_type}. Available: {list(extractors.keys())}"
        )

    return extractors[extractor_type](**kwargs)
