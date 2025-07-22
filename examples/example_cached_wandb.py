#!/usr/bin/env python3
"""
Example showing how to use feature caching and W&B logging with the attribute probes framework.
"""

import logging
import os
from experiment_runner import ExperimentRunner, PRESET_CONFIGS
from feature_cache import FeatureCacheManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def example_with_caching_and_wandb():
    """
    Demonstrate feature caching and W&B logging.
    """
    print("=== Example with Feature Caching and W&B Logging ===")
    
    # Initialize experiment runner with caching and W&B enabled
    # Set use_wandb=True and provide your W&B entity if you want to log
    runner = ExperimentRunner(
        cache_dir="cached_features",
        use_wandb=False,  # Set to True to enable W&B logging
        wandb_project="attribute-probes-demo",
        wandb_entity=None  # Add your W&B username/team here
    )
    
    # Configuration for DINOv2 feature extraction
    extractor_config = {
        'extractor_type': 'dinov2',
        'model_name': 'facebook/dinov2-base',
        'layer': 'last',
        'batch_size': 32
    }
    
    probe_config = {
        'probe_type': 'logistic',
        'random_seed': 42
    }
    
    print("\n1. Running first experiment (will extract and cache features)...")
    results1 = runner.run_single_experiment(
        extractor_config=extractor_config,
        probe_config=probe_config,
        experiment_name="dinov2_base_first_run",
        tags=['example', 'first_run', 'dinov2']
    )
    
    cache_key1 = results1['cache_key']
    print(f"Features cached with key: {cache_key1}")
    print(f"Extraction time: {results1['timing']['feature_extraction_time']:.2f}s")
    
    print("\n2. Running second experiment (should load from cache)...")
    results2 = runner.run_single_experiment(
        extractor_config=extractor_config,
        probe_config={'probe_type': 'mlp'},  # Different probe type
        experiment_name="dinov2_base_second_run_mlp",
        tags=['example', 'second_run', 'dinov2', 'mlp']
    )
    
    cache_key2 = results2['cache_key']
    print(f"Features loaded from cache key: {cache_key2}")
    print(f"Extraction time: {results2['timing']['feature_extraction_time']:.2f}s")
    print(f"Cache hit: {cache_key1 == cache_key2}")
    
    return runner, cache_key1


def example_save_features_as_hf_dataset(runner, cache_key):
    """
    Save cached features as a HuggingFace dataset.
    """
    print("\n=== Saving Features as HuggingFace Dataset ===")
    
    # Save the cached features as a standalone HF dataset
    output_path = "dinov2_base_features_dataset"
    runner.save_features_as_hf_dataset(cache_key, output_path)
    
    # Load the saved dataset to verify
    from feature_cache import FeatureCacheManager
    cache_manager = FeatureCacheManager()
    features, labels = cache_manager.load_hf_dataset_features(output_path)
    
    print(f"Saved and loaded HF dataset:")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return features, labels


def example_cache_management(runner):
    """
    Demonstrate cache management features.
    """
    print("\n=== Cache Management ===")
    
    # Get cache info
    cache_info = runner.get_cache_info()
    print(f"Cache info:")
    print(f"  Number of cached extractions: {cache_info['num_cached_extractions']}")
    print(f"  Total cache size: {cache_info['total_cache_size_mb']:.2f} MB")
    print(f"  Cache directory: {cache_info['cache_directory']}")
    
    # List all cached features
    cached_features = runner.list_cached_features()
    print(f"\nCached feature extractions:")
    for cache_key, metadata in cached_features.items():
        extractor = metadata['extractor_config']
        feature_shape = metadata['feature_shape']
        print(f"  {cache_key}: {extractor['extractor_type']} {extractor['model_name']} -> {feature_shape}")


def example_model_comparison_with_caching():
    """
    Run model comparison that benefits from caching.
    """
    print("\n=== Model Comparison with Feature Caching ===")
    
    runner = ExperimentRunner(
        cache_dir="cached_features",
        use_wandb=False,  # Set to True to enable W&B
        wandb_project="model-comparison-demo"
    )
    
    # Compare different DINOv2 sizes - features will be cached separately
    models_config = [
        {
            'extractor_type': 'dinov2',
            'model_name': 'facebook/dinov2-small',
            'layer': 'last',
            'batch_size': 64
        },
        {
            'extractor_type': 'dinov2', 
            'model_name': 'facebook/dinov2-base',
            'layer': 'last',
            'batch_size': 32
        }
    ]
    
    probe_config = {'probe_type': 'logistic'}
    
    # First run - will extract and cache features for each model
    print("\nFirst comparison run (extracting features)...")
    results1 = runner.run_model_comparison(
        models_config=models_config,
        probe_config=probe_config,
        experiment_prefix="dinov2_size_comparison",
        tags=['comparison', 'dinov2', 'size']
    )
    
    # Second run with different probe - will use cached features
    print("\nSecond comparison run (using cached features)...")
    results2 = runner.run_model_comparison(
        models_config=models_config,
        probe_config={'probe_type': 'mlp'},
        experiment_prefix="dinov2_size_comparison_mlp",
        tags=['comparison', 'dinov2', 'size', 'mlp']
    )
    
    # Print timing comparison
    print("\nTiming comparison:")
    for model_name in results1['comparison_summary'].keys():
        time1 = results1['individual_experiments'][model_name]['timing']['feature_extraction_time']
        time2 = results2['individual_experiments'][model_name]['timing']['feature_extraction_time']
        print(f"{model_name}: {time1:.2f}s -> {time2:.2f}s (speedup: {time1/time2:.1f}x)")
    
    return runner


def example_layer_analysis_with_shared_cache():
    """
    Layer analysis that shares feature cache when possible.
    """
    print("\n=== Layer Analysis with Shared Caching ===")
    
    runner = ExperimentRunner(
        cache_dir="cached_features",
        use_wandb=False,
        wandb_project="layer-analysis-demo"
    )
    
    # Base model configuration
    base_model_config = {
        'extractor_type': 'dinov2',
        'model_name': 'facebook/dinov2-base'
    }
    
    # Analyze different layers - each will be cached separately
    layers = ['last', 6, 8, 10]
    
    results = runner.run_layer_analysis(
        base_model_config=base_model_config,
        layers=layers,
        probe_config={'probe_type': 'logistic'},
        experiment_prefix="dinov2_layer_analysis"
    )
    
    # Show how different layers got different cache keys
    print("\nCache keys by layer:")
    for layer in layers:
        layer_key = f'layer_{layer}'
        if layer_key in results['individual_experiments']:
            cache_key = results['individual_experiments'][layer_key]['cache_key']
            print(f"  Layer {layer}: {cache_key[:8]}...")
    
    return runner


def example_wandb_sweep_config():
    """
    Example W&B sweep configuration for hyperparameter tuning.
    """
    print("\n=== W&B Sweep Configuration Example ===")
    
    # Example sweep configuration for probe hyperparameters
    sweep_config = {
        'method': 'bayes',
        'name': 'attribute_probe_hyperparameter_sweep',
        'metric': {
            'name': 'summary/f1_mean',
            'goal': 'maximize'
        },
        'parameters': {
            'probe_type': {
                'values': ['logistic', 'mlp']
            },
            'hidden_dims': {
                'values': [[256], [512], [256, 128], [512, 256]]
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 0.0001,
                'max': 0.01
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.5
            }
        }
    }
    
    print("Example sweep configuration:")
    print(f"  Method: {sweep_config['method']}")
    print(f"  Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print(f"  Parameters: {list(sweep_config['parameters'].keys())}")
    
    # To actually create and run the sweep:
    # runner = ExperimentRunner(use_wandb=True, wandb_project="probe-hyperparameter-sweep")
    # sweep_id = runner.wandb_tracker.create_sweep_config(sweep_config)
    # wandb.agent(sweep_id, function=your_sweep_function)
    
    return sweep_config


if __name__ == '__main__':
    print("=== Feature Caching and W&B Integration Examples ===")
    
    # Basic caching and logging
    runner, cache_key = example_with_caching_and_wandb()
    
    # Save features as HF dataset
    example_save_features_as_hf_dataset(runner, cache_key)
    
    # Cache management
    example_cache_management(runner)
    
    # Model comparison with caching
    example_model_comparison_with_caching()
    
    # Layer analysis with caching
    example_layer_analysis_with_shared_cache()
    
    # W&B sweep example
    example_wandb_sweep_config()
    
    print("\n=== All Examples Completed ===")
    print("To enable W&B logging:")
    print("1. Set use_wandb=True in ExperimentRunner")
    print("2. Add your W&B entity/username")
    print("3. Install wandb: pip install wandb")
    print("4. Login: wandb login")
    
    print("\nFeature caching benefits:")
    print("- Automatic caching using HuggingFace datasets")
    print("- Significant speedup on repeated experiments")
    print("- Easy export to standalone datasets")
    print("- Automatic cache key generation based on configs")