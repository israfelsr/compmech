import logging
from experiment_runner import ExperimentRunner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_baseline_replication(concept_file='mcrae-x-things.json', 
                           attribute_file='mcrae-x-things-taxonomy.json',
                           image_dir='images',
                           model_name='facebook/dinov2-base'):
    """
    Run the baseline replication using the modular experiment runner.
    
    Args:
        concept_file: Path to concept-attribute mappings
        attribute_file: Path to attribute taxonomy mappings  
        image_dir: Directory containing images
        model_name: Pretrained model to use for feature extraction
    """
    
    # Setup experiment runner
    runner = ExperimentRunner(
        concept_file=concept_file,
        attribute_file=attribute_file,
        image_dir=image_dir
    )
    
    # Configure feature extractor (DINOv2 with last layer)
    extractor_config = {
        'extractor_type': 'dinov2',
        'model_name': model_name,
        'layer': 'last',
        'batch_size': 32
    }
    
    # Configure probe (Logistic regression)
    probe_config = {
        'probe_type': 'logistic',
        'random_seed': 42
    }
    
    # Run baseline experiment
    results = runner.run_single_experiment(
        extractor_config=extractor_config,
        probe_config=probe_config,
        experiment_name='baseline_replication',
        cv_folds=5,
        n_repeats=2
    )
    
    # Print results in the same format as original
    summary = results['summary']
    logging.info("=== BASELINE REPLICATION RESULTS ===")
    logging.info(f"Tested {summary['n_attributes_tested']} attributes")
    logging.info(f"Mean F1 across attributes: {summary['mean_f1_across_attributes']:.4f} ± {summary['std_f1_across_attributes']:.4f}")
    logging.info(f"Median F1: {summary['median_f1_across_attributes']:.4f}")
    logging.info(f"F1 range: [{summary['min_f1']:.4f}, {summary['max_f1']:.4f}]")
    
    # Save results
    runner.save_results(results, 'baseline_replication_results.json')
    
    return results


if __name__ == '__main__':
    # Run baseline replication
    results = run_baseline_replication()