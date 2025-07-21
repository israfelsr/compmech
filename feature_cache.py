import os
import hashlib
import json
import numpy as np
import torch
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
from datasets import Dataset, load_from_disk, save_to_disk
from feature_extractors import BaseFeatureExtractor


class FeatureCache:
    """
    Feature caching system using HuggingFace datasets for efficient storage and retrieval.
    Automatically handles feature extraction, caching, and loading.
    """
    
    def __init__(self, cache_dir: str = "cached_features"):
        """
        Initialize feature cache.
        
        Args:
            cache_dir: Directory to store cached features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Metadata file to track cached extractions
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        logging.info(f"Feature cache initialized at {self.cache_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_cache_key(self, extractor_config: Dict, dataset_config: Dict) -> str:
        """Generate unique cache key based on extractor and dataset configuration."""
        # Create a string representation of all relevant configurations
        cache_string = json.dumps({
            'extractor_config': extractor_config,
            'dataset_config': dataset_config
        }, sort_keys=True)
        
        # Generate hash
        return hashlib.md5(cache_string.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for cached dataset."""
        return self.cache_dir / f"features_{cache_key}"
    
    def extract_or_load_features(self, extractor: BaseFeatureExtractor, 
                                dataset, extractor_config: Dict,
                                force_recompute: bool = False) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Extract features or load from cache if available.
        
        Args:
            extractor: Feature extractor instance
            dataset: Dataset to extract features from
            extractor_config: Configuration used for the extractor
            force_recompute: Force recomputation even if cached version exists
            
        Returns:
            tuple: (features, labels, cache_key)
        """
        # Create dataset configuration for cache key
        dataset_config = {
            'dataset_length': len(dataset),
            'num_attributes': getattr(dataset, 'num_attributes', None),
            'concept_file': getattr(dataset, 'concept_file', None),
            'attribute_file': getattr(dataset, 'attribute_file', None),
        }
        
        cache_key = self._generate_cache_key(extractor_config, dataset_config)
        cache_path = self._get_cache_path(cache_key)
        
        # Check if cached version exists and is not forced to recompute
        if not force_recompute and cache_path.exists() and cache_key in self.metadata:
            logging.info(f"Loading cached features from {cache_path}")
            return self._load_cached_features(cache_key)
        
        # Extract features
        logging.info(f"Extracting features (cache key: {cache_key})")
        features, labels = extractor.extract_features(dataset)
        
        # Cache the features
        self._save_cached_features(features, labels, cache_key, extractor_config, dataset_config)
        
        return features, labels, cache_key
    
    def _save_cached_features(self, features: np.ndarray, labels: np.ndarray,
                             cache_key: str, extractor_config: Dict, dataset_config: Dict):
        """Save extracted features to cache."""
        cache_path = self._get_cache_path(cache_key)
        
        # Convert to HuggingFace dataset format
        # Store features and labels as lists for HF dataset compatibility
        dataset_dict = {
            'features': features.tolist(),
            'labels': labels.tolist(),
            'feature_shape': list(features.shape),
            'label_shape': list(labels.shape)
        }
        
        # Create HuggingFace dataset
        hf_dataset = Dataset.from_dict(dataset_dict)
        
        # Save dataset
        hf_dataset.save_to_disk(str(cache_path))
        
        # Update metadata
        self.metadata[cache_key] = {
            'extractor_config': extractor_config,
            'dataset_config': dataset_config,
            'feature_shape': list(features.shape),
            'label_shape': list(labels.shape),
            'cache_path': str(cache_path),
            'created_at': datetime.now().isoformat()
        }
        
        self._save_metadata()
        logging.info(f"Features cached to {cache_path}")
    
    def _load_cached_features(self, cache_key: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """Load features from cache."""
        cache_path = self._get_cache_path(cache_key)
        
        # Load HuggingFace dataset
        hf_dataset = Dataset.load_from_disk(str(cache_path))
        
        # Convert back to numpy arrays
        features = np.array(hf_dataset['features'])
        labels = np.array(hf_dataset['labels'])
        
        # Reshape to original dimensions
        feature_shape = tuple(hf_dataset['feature_shape'][0])
        label_shape = tuple(hf_dataset['label_shape'][0])
        
        features = features.reshape(feature_shape)
        labels = labels.reshape(label_shape)
        
        logging.info(f"Loaded cached features: {features.shape}, labels: {labels.shape}")
        
        return features, labels, cache_key
    
    def load_features_by_key(self, cache_key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load features directly by cache key."""
        if cache_key not in self.metadata:
            raise ValueError(f"Cache key {cache_key} not found")
        
        features, labels, _ = self._load_cached_features(cache_key)
        return features, labels
    
    def list_cached_features(self) -> Dict:
        """List all cached features with their metadata."""
        return self.metadata.copy()
    
    def clear_cache(self, cache_key: Optional[str] = None):
        """Clear cache entries."""
        if cache_key:
            # Clear specific cache entry
            if cache_key in self.metadata:
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    import shutil
                    shutil.rmtree(cache_path)
                del self.metadata[cache_key]
                self._save_metadata()
                logging.info(f"Cleared cache for key: {cache_key}")
        else:
            # Clear all cache
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.metadata = {}
            self._save_metadata()
            logging.info("Cleared all cached features")
    
    def get_cache_info(self) -> Dict:
        """Get information about cache usage."""
        total_size = 0
        num_entries = len(self.metadata)
        
        for cache_key, metadata in self.metadata.items():
            cache_path = Path(metadata['cache_path'])
            if cache_path.exists():
                total_size += sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
        
        return {
            'num_cached_extractions': num_entries,
            'total_cache_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir),
            'cached_keys': list(self.metadata.keys())
        }


class FeatureCacheManager:
    """
    High-level manager for feature caching that integrates with experiment workflows.
    """
    
    def __init__(self, cache_dir: str = "cached_features"):
        self.cache = FeatureCache(cache_dir)
    
    def get_or_extract_features(self, extractor_config: Dict, dataset,
                               force_recompute: bool = False) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get features from cache or extract them if not cached.
        
        Args:
            extractor_config: Configuration for feature extractor
            dataset: Dataset to extract features from
            force_recompute: Force recomputation
            
        Returns:
            tuple: (features, labels, cache_key)
        """
        from feature_extractors import get_feature_extractor
        
        # Create extractor
        extractor = get_feature_extractor(**extractor_config)
        
        # Extract or load features
        return self.cache.extract_or_load_features(
            extractor, dataset, extractor_config, force_recompute
        )
    
    def save_features_as_hf_dataset(self, cache_key: str, output_path: str,
                                   include_metadata: bool = True):
        """
        Save cached features as a standalone HuggingFace dataset.
        
        Args:
            cache_key: Cache key of features to save
            output_path: Path to save the dataset
            include_metadata: Whether to include extraction metadata
        """
        features, labels = self.cache.load_features_by_key(cache_key)
        
        # Create dataset dictionary
        dataset_dict = {
            'features': features.tolist(),
            'labels': labels.tolist(),
        }
        
        if include_metadata:
            metadata = self.cache.metadata[cache_key]
            dataset_dict['metadata'] = [metadata] * len(features)  # Repeat for each sample
        
        # Create and save HuggingFace dataset
        hf_dataset = Dataset.from_dict(dataset_dict)
        hf_dataset.save_to_disk(output_path)
        
        logging.info(f"Saved features as HF dataset to {output_path}")
    
    def load_hf_dataset_features(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features from a saved HuggingFace dataset.
        
        Args:
            dataset_path: Path to saved HF dataset
            
        Returns:
            tuple: (features, labels)
        """
        hf_dataset = Dataset.load_from_disk(dataset_path)
        
        features = np.array(hf_dataset['features'])
        labels = np.array(hf_dataset['labels'])
        
        return features, labels