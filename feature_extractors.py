import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import logging
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
    
    @abstractmethod
    def extract_features(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from dataset."""
        pass


class DINOv2FeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using DINOv2 models."""
    
    def __init__(self, model_name='facebook/dinov2-base', layer='last', 
                 batch_size=32, device='auto'):
        """
        Initialize DINOv2 feature extractor.
        
        Args:
            model_name: HuggingFace model name
            layer: Which layer to extract from ('last', 'intermediate', or layer index)
            batch_size: Batch size for inference
            device: Device to run on
        """
        super().__init__(device)
        self.model_name = model_name
        self.layer = layer
        self.batch_size = batch_size
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"Loaded {model_name} on {self.device}")
    
    def extract_features(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from dataset using DINOv2.
        
        Args:
            dataset: Dataset to extract features from
            
        Returns:
            tuple: (features, labels) as numpy arrays
        """
        logging.info(f"Extracting features from {self.model_name} ({self.layer} layer)...")
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                              shuffle=False, num_workers=0)
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting features")):
                images = images.to(self.device)
                
                # Forward pass through model
                outputs = self.model(images)
                
                # Extract features based on specified layer
                if self.layer == 'last':
                    # CLS token from last layer
                    features = outputs.last_hidden_state[:, 0]
                elif self.layer == 'intermediate':
                    # Average of intermediate layers
                    hidden_states = outputs.hidden_states
                    features = torch.stack(hidden_states).mean(dim=0)[:, 0]
                elif isinstance(self.layer, int):
                    # Specific layer
                    features = outputs.hidden_states[self.layer][:, 0]
                else:
                    raise ValueError(f"Unknown layer specification: {self.layer}")
                
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
        
        features = torch.cat(all_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        logging.info(f"Extracted features shape: {features.shape}")
        logging.info(f"Labels shape: {labels.shape}")
        
        return features, labels


class CLIPFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using CLIP models."""
    
    def __init__(self, model_name='openai/clip-vit-base-patch32', 
                 batch_size=32, device='auto'):
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
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                              shuffle=False, num_workers=0)
        
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


class ResNetFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using ResNet models."""
    
    def __init__(self, model_name='resnet50', pretrained=True, 
                 layer='avgpool', batch_size=32, device='auto'):
        super().__init__(device)
        self.model_name = model_name
        self.layer = layer
        self.batch_size = batch_size
        
        import torchvision.models as models
        from torchvision import transforms
        
        # Load model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Register hook to extract features
        self.features = []
        if layer == 'avgpool':
            self.model.avgpool.register_forward_hook(self._get_features)
        
        logging.info(f"Loaded {model_name} on {self.device}")
    
    def _get_features(self, module, input, output):
        """Hook to capture features."""
        self.features.append(output.detach().cpu())
    
    def extract_features(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features using ResNet."""
        logging.info(f"Extracting {self.model_name} features...")
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                              shuffle=False, num_workers=0)
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Extracting {self.model_name} features"):
                images = self.transform(images).to(self.device)
                
                self.features = []  # Clear previous features
                _ = self.model(images)  # Forward pass triggers hook
                
                features = self.features[0].flatten(1)  # Flatten spatial dimensions
                all_features.append(features)
                all_labels.append(labels.cpu())
        
        features = torch.cat(all_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        logging.info(f"Extracted {self.model_name} features shape: {features.shape}")
        
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
        
        logging.info(f"Combined features from {len(self.layers)} layers: {combined_features.shape}")
        
        return combined_features, labels


def get_feature_extractor(extractor_type: str, **kwargs) -> BaseFeatureExtractor:
    """Factory function to create feature extractors."""
    extractors = {
        'dinov2': DINOv2FeatureExtractor,
        'clip': CLIPFeatureExtractor,
        'resnet': ResNetFeatureExtractor,
    }
    
    if extractor_type not in extractors:
        raise ValueError(f"Unknown extractor type: {extractor_type}. Available: {list(extractors.keys())}")
    
    return extractors[extractor_type](**kwargs)