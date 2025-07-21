#!/usr/bin/env python3
"""
Example experiments using the modular AttributeProbes framework.
This demonstrates how to easily run different types of experiments.
"""

import logging
from experiment_runner import ExperimentRunner, PRESET_CONFIGS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def example_model_comparison():
    """Compare different vision models on attribute prediction."""
    print("=== Running Model Comparison ===")
    
    runner = ExperimentRunner()
    
    # Use preset vision models configuration
    models_config = PRESET_CONFIGS['vision_models']
    
    # Use logistic regression probes
    probe_config = {'probe_type': 'logistic'}
    
    results = runner.run_model_comparison(
        models_config=models_config,
        probe_config=probe_config,
        experiment_prefix="vision_model_comparison"
    )
    
    runner.save_results(results, "model_comparison_results.json")
    
    # Print comparison summary
    print("\nModel Comparison Results:")
    for model_name, summary in results['comparison_summary'].items():
        print(f"{model_name}: F1 = {summary['mean_f1_across_attributes']:.4f} ± {summary['std_f1_across_attributes']:.4f}")


def example_probe_comparison():
    """Compare different probe types on the same features."""
    print("\n=== Running Probe Comparison ===")
    
    runner = ExperimentRunner()
    
    # Use DINOv2-base features
    extractor_config = {
        'extractor_type': 'dinov2',
        'model_name': 'facebook/dinov2-base',
        'layer': 'last'
    }
    
    # Compare different probe types
    probe_configs = PRESET_CONFIGS['probe_types']
    
    results = runner.run_probe_comparison(
        extractor_config=extractor_config,
        probe_configs=probe_configs,
        experiment_prefix="probe_comparison"
    )
    
    runner.save_results(results, "probe_comparison_results.json")
    
    # Print comparison summary
    print("\nProbe Comparison Results:")
    for probe_name, summary in results['comparison_summary'].items():
        print(f"{probe_name}: F1 = {summary['mean_f1_across_attributes']:.4f} ± {summary['std_f1_across_attributes']:.4f}")


def example_layer_analysis():
    """Analyze different layers of DINOv2."""
    print("\n=== Running Layer Analysis ===")
    
    runner = ExperimentRunner()
    
    # Base DINOv2 configuration
    base_model_config = {
        'extractor_type': 'dinov2',
        'model_name': 'facebook/dinov2-base'
    }
    
    # Use logistic regression probes
    probe_config = {'probe_type': 'logistic'}
    
    # Analyze different layers
    layers = PRESET_CONFIGS['dinov2_layers']
    
    results = runner.run_layer_analysis(
        base_model_config=base_model_config,
        layers=layers,
        probe_config=probe_config,
        experiment_prefix="dinov2_layer_analysis"
    )
    
    runner.save_results(results, "layer_analysis_results.json")
    
    # Print layer comparison
    print("\nLayer Analysis Results:")
    for layer_name, summary in results['layer_summary'].items():
        print(f"{layer_name}: F1 = {summary['mean_f1_across_attributes']:.4f} ± {summary['std_f1_across_attributes']:.4f}")


def example_attribute_subset_analysis():
    """Analyze performance on different types of attributes."""
    print("\n=== Running Attribute Subset Analysis ===")
    
    runner = ExperimentRunner()
    
    # Example attribute groups (you would define these based on your taxonomy)
    attribute_groups = {
        'visual_attributes': list(range(0, 50)),    # First 50 attributes
        'functional_attributes': list(range(50, 100)),  # Next 50 attributes
        'categorical_attributes': list(range(100, 150))  # Next 50 attributes
    }
    
    extractor_config = {
        'extractor_type': 'dinov2',
        'model_name': 'facebook/dinov2-base',
        'layer': 'last'
    }
    
    probe_config = {'probe_type': 'logistic'}
    
    results = runner.run_attribute_subset_analysis(
        extractor_config=extractor_config,
        probe_config=probe_config,
        attribute_groups=attribute_groups,
        experiment_prefix="attribute_subset_analysis"
    )
    
    runner.save_results(results, "attribute_subset_results.json")
    
    # Print group comparison
    print("\nAttribute Subset Results:")
    for group_name, summary in results['group_summary'].items():
        print(f"{group_name}: F1 = {summary['mean_f1_across_attributes']:.4f} ± {summary['std_f1_across_attributes']:.4f}")


def example_custom_experiment():
    """Run a custom experiment with specific configuration."""
    print("\n=== Running Custom Experiment ===")
    
    runner = ExperimentRunner()
    
    # Custom extractor configuration
    extractor_config = {
        'extractor_type': 'dinov2',
        'model_name': 'facebook/dinov2-large',  # Use large model
        'layer': 10,  # Specific layer
        'batch_size': 16  # Smaller batch size for large model
    }
    
    # Custom probe configuration with MLP
    probe_config = {
        'probe_type': 'mlp',  # Use MLP instead of logistic regression
        'random_seed': 42
    }
    
    results = runner.run_single_experiment(
        extractor_config=extractor_config,
        probe_config=probe_config,
        experiment_name="custom_dinov2_large_mlp",
        cv_folds=5,
        n_repeats=2
    )
    
    runner.save_results(results, "custom_experiment_results.json")
    
    summary = results['summary']
    print(f"\nCustom Experiment Results:")
    print(f"Mean F1: {summary['mean_f1_across_attributes']:.4f} ± {summary['std_f1_across_attributes']:.4f}")


def example_torch_probe_experiment():
    """Example using PyTorch-based MLP probes."""
    print("\n=== Running PyTorch MLP Probe Experiment ===")
    
    runner = ExperimentRunner()
    
    extractor_config = {
        'extractor_type': 'dinov2',
        'model_name': 'facebook/dinov2-base',
        'layer': 'last'
    }
    
    # PyTorch MLP probe configuration
    probe_config = {
        'type': 'torch',  # Use TorchAttributeProbes
        'hidden_dims': [512, 256, 128],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 50
    }
    
    results = runner.run_single_experiment(
        extractor_config=extractor_config,
        probe_config=probe_config,
        experiment_name="torch_mlp_probe",
        cv_folds=5
    )
    
    runner.save_results(results, "torch_mlp_results.json")
    
    summary = results['summary']
    print(f"\nTorch MLP Probe Results:")
    print(f"Mean F1: {summary['mean_f1_across_attributes']:.4f} ± {summary['std_f1_across_attributes']:.4f}")


if __name__ == '__main__':
    # Run different example experiments
    
    # Basic comparisons
    example_model_comparison()
    example_probe_comparison() 
    
    # Advanced analyses
    example_layer_analysis()
    example_attribute_subset_analysis()
    
    # Custom experiments
    example_custom_experiment()
    example_torch_probe_experiment()
    
    print("\n=== All Example Experiments Completed ===")
    print("Check the 'results/' directory for saved results.")