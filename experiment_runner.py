import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

from probes import AttributeProbes, TorchAttributeProbes
from feature_extractors import get_feature_extractor, BaseFeatureExtractor
from data import ConceptAttributesDataset


class ExperimentRunner:
    """
    Main experiment runner that coordinates feature extraction and probe training
    across different models and configurations.
    """
    
    def __init__(self, concept_file='mcrae-x-things.json', 
                 attribute_file='mcrae-x-things-taxonomy.json',
                 image_dir='images', results_dir='results'):
        """
        Initialize the experiment runner.
        
        Args:
            concept_file: Path to concept-attribute mappings
            attribute_file: Path to attribute taxonomy mappings  
            image_dir: Directory containing images
            results_dir: Directory to save results
        """
        self.concept_file = concept_file
        self.attribute_file = attribute_file
        self.image_dir = image_dir
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load dataset
        self.dataset = ConceptAttributesDataset(
            concept_file=concept_file,
            attribute_file=attribute_file, 
            image_dir=image_dir
        )
        
        logging.info(f"Loaded dataset with {len(self.dataset)} concepts and {self.dataset.num_attributes} attributes")
        
    def run_single_experiment(self, extractor_config: Dict, probe_config: Dict,
                            experiment_name: str, cv_folds: int = 5, 
                            n_repeats: int = 2) -> Dict:
        """
        Run a single experiment with specified feature extractor and probe configuration.
        
        Args:
            extractor_config: Configuration for feature extractor
            probe_config: Configuration for probe
            experiment_name: Name for this experiment
            cv_folds: Number of CV folds
            n_repeats: Number of CV repeats
            
        Returns:
            dict: Experiment results
        """
        logging.info(f"Starting experiment: {experiment_name}")
        
        # Create feature extractor
        extractor = get_feature_extractor(**extractor_config)
        
        # Extract features
        features, labels = extractor.extract_features(self.dataset)
        
        # Create probe
        if probe_config.get('type') == 'torch':
            probe = TorchAttributeProbes(dataset=self.dataset, **probe_config)
            import torch
            features_tensor = torch.FloatTensor(features)
            labels_tensor = torch.FloatTensor(labels)
            results = probe.train_all_probes(features_tensor, labels_tensor, cv_folds)
        else:
            probe = AttributeProbes(dataset=self.dataset, **probe_config)
            results = probe.train_all_probes(features, labels, cv_folds, n_repeats)
        
        # Add experiment metadata
        results['experiment_name'] = experiment_name
        results['extractor_config'] = extractor_config
        results['probe_config'] = probe_config
        results['cv_folds'] = cv_folds
        results['n_repeats'] = n_repeats
        results['timestamp'] = datetime.now().isoformat()
        
        return results
    
    def run_model_comparison(self, models_config: List[Dict], probe_config: Dict,
                           experiment_prefix: str = "model_comparison") -> Dict:
        """
        Compare different models using the same probe configuration.
        
        Args:
            models_config: List of feature extractor configurations
            probe_config: Probe configuration to use for all models
            experiment_prefix: Prefix for experiment names
            
        Returns:
            dict: Comparison results across all models
        """
        logging.info(f"Running model comparison with {len(models_config)} models")
        
        all_results = {}
        comparison_summary = {}
        
        for i, model_config in enumerate(models_config):
            model_name = model_config.get('model_name', f'model_{i}')
            experiment_name = f"{experiment_prefix}_{model_name}"
            
            try:
                results = self.run_single_experiment(
                    extractor_config=model_config,
                    probe_config=probe_config,
                    experiment_name=experiment_name
                )
                
                all_results[model_name] = results
                comparison_summary[model_name] = results['summary']
                
                logging.info(f"Model {model_name}: Mean F1 = {results['summary']['mean_f1_across_attributes']:.4f}")
                
            except Exception as e:
                logging.error(f"Error running experiment for {model_name}: {str(e)}")
                continue
        
        return {
            'individual_experiments': all_results,
            'comparison_summary': comparison_summary,
            'experiment_type': 'model_comparison',
            'timestamp': datetime.now().isoformat()
        }
    
    def run_probe_comparison(self, extractor_config: Dict, probe_configs: List[Dict],
                           experiment_prefix: str = "probe_comparison") -> Dict:
        """
        Compare different probe types using the same feature extractor.
        
        Args:
            extractor_config: Feature extractor configuration
            probe_configs: List of probe configurations
            experiment_prefix: Prefix for experiment names
            
        Returns:
            dict: Comparison results across all probe types
        """
        logging.info(f"Running probe comparison with {len(probe_configs)} probe types")
        
        all_results = {}
        comparison_summary = {}
        
        for i, probe_config in enumerate(probe_configs):
            probe_name = probe_config.get('probe_type', f'probe_{i}')
            experiment_name = f"{experiment_prefix}_{probe_name}"
            
            try:
                results = self.run_single_experiment(
                    extractor_config=extractor_config,
                    probe_config=probe_config,
                    experiment_name=experiment_name
                )
                
                all_results[probe_name] = results
                comparison_summary[probe_name] = results['summary']
                
                logging.info(f"Probe {probe_name}: Mean F1 = {results['summary']['mean_f1_across_attributes']:.4f}")
                
            except Exception as e:
                logging.error(f"Error running experiment for {probe_name}: {str(e)}")
                continue
        
        return {
            'individual_experiments': all_results,
            'comparison_summary': comparison_summary,
            'experiment_type': 'probe_comparison',
            'timestamp': datetime.now().isoformat()
        }
    
    def run_layer_analysis(self, base_model_config: Dict, layers: List,
                          probe_config: Dict, experiment_prefix: str = "layer_analysis") -> Dict:
        """
        Analyze different layers of a model.
        
        Args:
            base_model_config: Base model configuration
            layers: List of layers to analyze
            probe_config: Probe configuration
            experiment_prefix: Prefix for experiment names
            
        Returns:
            dict: Layer analysis results
        """
        logging.info(f"Running layer analysis for {len(layers)} layers")
        
        all_results = {}
        layer_summary = {}
        
        for layer in layers:
            model_config = base_model_config.copy()
            model_config['layer'] = layer
            experiment_name = f"{experiment_prefix}_layer_{layer}"
            
            try:
                results = self.run_single_experiment(
                    extractor_config=model_config,
                    probe_config=probe_config,
                    experiment_name=experiment_name
                )
                
                all_results[f'layer_{layer}'] = results
                layer_summary[f'layer_{layer}'] = results['summary']
                
                logging.info(f"Layer {layer}: Mean F1 = {results['summary']['mean_f1_across_attributes']:.4f}")
                
            except Exception as e:
                logging.error(f"Error running experiment for layer {layer}: {str(e)}")
                continue
        
        return {
            'individual_experiments': all_results,
            'layer_summary': layer_summary,
            'experiment_type': 'layer_analysis',
            'analyzed_layers': layers,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_attribute_subset_analysis(self, extractor_config: Dict, probe_config: Dict,
                                    attribute_groups: Dict[str, List[int]],
                                    experiment_prefix: str = "attribute_analysis") -> Dict:
        """
        Analyze performance on different subsets of attributes.
        
        Args:
            extractor_config: Feature extractor configuration
            probe_config: Probe configuration
            attribute_groups: Dict mapping group names to attribute indices
            experiment_prefix: Prefix for experiment names
            
        Returns:
            dict: Attribute subset analysis results
        """
        logging.info(f"Running attribute subset analysis for {len(attribute_groups)} groups")
        
        # First extract features once
        extractor = get_feature_extractor(**extractor_config)
        features, labels = extractor.extract_features(self.dataset)
        
        all_results = {}
        group_summary = {}
        
        for group_name, attribute_indices in attribute_groups.items():
            experiment_name = f"{experiment_prefix}_{group_name}"
            
            try:
                # Create probe
                probe = AttributeProbes(dataset=self.dataset, **probe_config)
                
                # Evaluate specific attributes
                results = probe.evaluate_specific_attributes(
                    features, labels, attribute_indices
                )
                
                # Add metadata
                results['experiment_name'] = experiment_name
                results['attribute_group'] = group_name
                results['extractor_config'] = extractor_config
                results['probe_config'] = probe_config
                results['timestamp'] = datetime.now().isoformat()
                
                all_results[group_name] = results
                group_summary[group_name] = results['summary']
                
                logging.info(f"Group {group_name}: Mean F1 = {results['summary']['mean_f1_across_attributes']:.4f}")
                
            except Exception as e:
                logging.error(f"Error running experiment for group {group_name}: {str(e)}")
                continue
        
        return {
            'individual_experiments': all_results,
            'group_summary': group_summary,
            'experiment_type': 'attribute_subset_analysis',
            'attribute_groups': attribute_groups,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file."""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {filepath}")
    
    def load_results(self, filename: str) -> Dict:
        """Load results from JSON file."""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'r') as f:
            results = json.load(f)
        logging.info(f"Results loaded from {filepath}")
        return results


# Predefined experiment configurations
PRESET_CONFIGS = {
    'dinov2_models': [
        {'extractor_type': 'dinov2', 'model_name': 'facebook/dinov2-small'},
        {'extractor_type': 'dinov2', 'model_name': 'facebook/dinov2-base'},
        {'extractor_type': 'dinov2', 'model_name': 'facebook/dinov2-large'},
    ],
    
    'vision_models': [
        {'extractor_type': 'dinov2', 'model_name': 'facebook/dinov2-base'},
        {'extractor_type': 'clip', 'model_name': 'openai/clip-vit-base-patch32'},
        {'extractor_type': 'resnet', 'model_name': 'resnet50'},
    ],
    
    'probe_types': [
        {'probe_type': 'logistic'},
        {'probe_type': 'mlp'},
        {'type': 'torch', 'hidden_dims': [512, 256]},
    ],
    
    'dinov2_layers': ['last', 'intermediate', 6, 8, 10, 11],
}