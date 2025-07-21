import wandb
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class WandbLogger:
    """
    Weights & Biases integration for logging attribute probe experiments.
    """
    
    def __init__(self, project_name: str = "attribute-probes", 
                 entity: Optional[str] = None,
                 enabled: bool = True):
        """
        Initialize wandb logger.
        
        Args:
            project_name: W&B project name
            entity: W&B entity (username or team)
            enabled: Whether to actually log to wandb
        """
        self.project_name = project_name
        self.entity = entity
        self.enabled = enabled
        self.run = None
        
        if not enabled:
            logging.info("W&B logging disabled")
    
    def init_run(self, experiment_name: str, config: Dict, 
                 tags: Optional[List[str]] = None, 
                 notes: Optional[str] = None):
        """
        Initialize a new W&B run.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary to log
            tags: List of tags for the run
            notes: Notes about the experiment
        """
        if not self.enabled:
            return
        
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=experiment_name,
            config=config,
            tags=tags or [],
            notes=notes,
            reinit=True
        )
        
        logging.info(f"Initialized W&B run: {experiment_name}")
    
    def log_experiment_config(self, extractor_config: Dict, probe_config: Dict,
                             cv_config: Dict, dataset_info: Dict):
        """
        Log experiment configuration.
        
        Args:
            extractor_config: Feature extractor configuration
            probe_config: Probe configuration
            cv_config: Cross-validation configuration
            dataset_info: Dataset information
        """
        if not self.enabled or not self.run:
            return
        
        # Update config with all experimental details
        full_config = {
            'extractor': extractor_config,
            'probe': probe_config,
            'cross_validation': cv_config,
            'dataset': dataset_info,
            'timestamp': datetime.now().isoformat()
        }
        
        wandb.config.update(full_config)
    
    def log_feature_extraction_info(self, features_shape: tuple, labels_shape: tuple,
                                   extraction_time: float, cache_key: Optional[str] = None):
        """
        Log feature extraction information.
        
        Args:
            features_shape: Shape of extracted features
            labels_shape: Shape of labels
            extraction_time: Time taken for feature extraction
            cache_key: Cache key if features were cached
        """
        if not self.enabled or not self.run:
            return
        
        wandb.log({
            'feature_extraction/features_shape': features_shape,
            'feature_extraction/labels_shape': labels_shape,
            'feature_extraction/n_samples': features_shape[0],
            'feature_extraction/n_features': features_shape[1] if len(features_shape) > 1 else 1,
            'feature_extraction/extraction_time_seconds': extraction_time,
            'feature_extraction/cache_key': cache_key or 'not_cached'
        })
    
    def log_attribute_results(self, attribute_results: List[Dict]):
        """
        Log individual attribute probe results.
        
        Args:
            attribute_results: List of results for each attribute
        """
        if not self.enabled or not self.run:
            return
        
        # Create a table for detailed per-attribute results
        columns = ['attribute_idx', 'attribute_name', 'mean_f1', 'std_f1', 
                  'mean_accuracy', 'std_accuracy', 'n_positive', 'n_total', 'class_balance']
        
        table_data = []
        for result in attribute_results:
            class_balance = result['n_positive'] / result['n_total'] if result['n_total'] > 0 else 0
            
            row = [
                result.get('attribute_idx', ''),
                result.get('attribute_name', f"attr_{result.get('attribute_idx', '')}"),
                result.get('mean_f1', 0),
                result.get('std_f1', 0),
                result.get('mean_accuracy', 0),
                result.get('std_accuracy', 0),
                result.get('n_positive', 0),
                result.get('n_total', 0),
                class_balance
            ]
            table_data.append(row)
        
        # Log as wandb table
        wandb.log({'attribute_results': wandb.Table(columns=columns, data=table_data)})
        
        # Log distributions
        f1_scores = [r.get('mean_f1', 0) for r in attribute_results]
        accuracies = [r.get('mean_accuracy', 0) for r in attribute_results]
        class_balances = [r['n_positive'] / r['n_total'] if r['n_total'] > 0 else 0 for r in attribute_results]
        
        wandb.log({
            'distributions/f1_histogram': wandb.Histogram(f1_scores),
            'distributions/accuracy_histogram': wandb.Histogram(accuracies),
            'distributions/class_balance_histogram': wandb.Histogram(class_balances)
        })
    
    def log_summary_metrics(self, summary: Dict):
        """
        Log summary metrics across all attributes.
        
        Args:
            summary: Summary statistics dictionary
        """
        if not self.enabled or not self.run:
            return
        
        metrics_to_log = {}
        
        # Main performance metrics
        for metric in ['f1', 'accuracy', 'precision', 'recall']:
            if f'mean_{metric}_across_attributes' in summary:
                metrics_to_log[f'summary/{metric}_mean'] = summary[f'mean_{metric}_across_attributes']
                metrics_to_log[f'summary/{metric}_std'] = summary.get(f'std_{metric}_across_attributes', 0)
                metrics_to_log[f'summary/{metric}_median'] = summary.get(f'median_{metric}_across_attributes', 0)
                metrics_to_log[f'summary/{metric}_min'] = summary.get(f'min_{metric}', 0)
                metrics_to_log[f'summary/{metric}_max'] = summary.get(f'max_{metric}', 0)
        
        # Number of attributes tested
        if 'n_attributes_tested' in summary:
            metrics_to_log['summary/n_attributes_tested'] = summary['n_attributes_tested']
        
        wandb.log(metrics_to_log)
    
    def log_comparison_results(self, comparison_data: Dict, comparison_type: str):
        """
        Log results from comparison experiments (model comparison, probe comparison, etc.).
        
        Args:
            comparison_data: Dictionary with comparison results
            comparison_type: Type of comparison ('models', 'probes', 'layers', etc.)
        """
        if not self.enabled or not self.run:
            return
        
        # Create comparison table
        columns = ['name', 'mean_f1', 'std_f1', 'mean_accuracy', 'std_accuracy', 'n_attributes']
        table_data = []
        
        f1_scores = []
        names = []
        
        for name, summary in comparison_data.items():
            f1_mean = summary.get('mean_f1_across_attributes', 0)
            f1_std = summary.get('std_f1_across_attributes', 0)
            acc_mean = summary.get('mean_accuracy_across_attributes', 0)
            acc_std = summary.get('std_accuracy_across_attributes', 0)
            n_attrs = summary.get('n_attributes_tested', 0)
            
            table_data.append([name, f1_mean, f1_std, acc_mean, acc_std, n_attrs])
            f1_scores.append(f1_mean)
            names.append(name)
        
        # Log comparison table
        wandb.log({f'{comparison_type}_comparison': wandb.Table(columns=columns, data=table_data)})
        
        # Log bar chart for F1 scores
        wandb.log({f'{comparison_type}_f1_comparison': wandb.plot.bar(
            wandb.Table(data=list(zip(names, f1_scores)), columns=['name', 'f1_score']),
            'name', 'f1_score', title=f'{comparison_type.capitalize()} F1 Score Comparison'
        )})
    
    def log_layer_analysis(self, layer_results: Dict):
        """
        Log layer analysis results with special visualizations.
        
        Args:
            layer_results: Dictionary with layer analysis results
        """
        if not self.enabled or not self.run:
            return
        
        layers = []
        f1_scores = []
        
        for layer_name, summary in layer_results.items():
            layer_num = layer_name.replace('layer_', '')
            layers.append(layer_num)
            f1_scores.append(summary.get('mean_f1_across_attributes', 0))
        
        # Create line plot for layer performance
        data = [[layer, f1] for layer, f1 in zip(layers, f1_scores)]
        table = wandb.Table(data=data, columns=['layer', 'f1_score'])
        
        wandb.log({
            'layer_analysis/f1_by_layer': wandb.plot.line(
                table, 'layer', 'f1_score', title='F1 Score by Layer'
            )
        })
    
    def log_attribute_group_analysis(self, group_results: Dict):
        """
        Log attribute group analysis results.
        
        Args:
            group_results: Dictionary with group analysis results
        """
        if not self.enabled or not self.run:
            return
        
        groups = []
        f1_scores = []
        n_attributes = []
        
        for group_name, summary in group_results.items():
            groups.append(group_name)
            f1_scores.append(summary.get('mean_f1_across_attributes', 0))
            n_attributes.append(summary.get('n_attributes_tested', 0))
        
        # Log group comparison
        data = list(zip(groups, f1_scores, n_attributes))
        table = wandb.Table(data=data, columns=['group', 'f1_score', 'n_attributes'])
        
        wandb.log({
            'group_analysis/comparison_table': table,
            'group_analysis/f1_by_group': wandb.plot.bar(
                table, 'group', 'f1_score', title='F1 Score by Attribute Group'
            )
        })
    
    def log_training_progress(self, epoch: int, metrics: Dict):
        """
        Log training progress for PyTorch probes.
        
        Args:
            epoch: Training epoch
            metrics: Dictionary of metrics
        """
        if not self.enabled or not self.run:
            return
        
        log_dict = {}
        for key, value in metrics.items():
            log_dict[f'training/{key}'] = value
        
        log_dict['training/epoch'] = epoch
        wandb.log(log_dict)
    
    def log_hyperparameter_sweep_result(self, config: Dict, metrics: Dict):
        """
        Log results for hyperparameter sweeps.
        
        Args:
            config: Hyperparameter configuration
            metrics: Result metrics
        """
        if not self.enabled or not self.run:
            return
        
        # Log config as hyperparameters
        for key, value in config.items():
            wandb.log({f'hp/{key}': value})
        
        # Log primary metrics
        for key, value in metrics.items():
            wandb.log({f'result/{key}': value})
    
    def finish_run(self):
        """Finish the current W&B run."""
        if not self.enabled or not self.run:
            return
        
        wandb.finish()
        logging.info("Finished W&B run")


class WandbExperimentTracker:
    """
    Higher-level experiment tracker that automatically logs common experiment patterns.
    """
    
    def __init__(self, project_name: str = "attribute-probes", 
                 entity: Optional[str] = None, enabled: bool = True):
        self.logger = WandbLogger(project_name, entity, enabled)
        self.enabled = enabled
    
    def track_single_experiment(self, experiment_name: str, config: Dict, 
                               results: Dict, tags: Optional[List[str]] = None):
        """
        Track a complete single experiment.
        
        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
            results: Experiment results
            tags: Tags for the run
        """
        if not self.enabled:
            return
        
        # Initialize run
        self.logger.init_run(experiment_name, config, tags)
        
        # Log results
        if 'individual_results' in results:
            self.logger.log_attribute_results(results['individual_results'])
        
        if 'summary' in results:
            self.logger.log_summary_metrics(results['summary'])
        
        # Finish run
        self.logger.finish_run()
    
    def track_comparison_experiment(self, experiment_name: str, 
                                  comparison_results: Dict, comparison_type: str,
                                  tags: Optional[List[str]] = None):
        """
        Track a comparison experiment.
        
        Args:
            experiment_name: Name of the experiment
            comparison_results: Results from comparison
            comparison_type: Type of comparison
            tags: Tags for the run
        """
        if not self.enabled:
            return
        
        config = {
            'experiment_type': comparison_type,
            'n_comparisons': len(comparison_results.get('comparison_summary', {}))
        }
        
        self.logger.init_run(experiment_name, config, tags)
        
        if 'comparison_summary' in comparison_results:
            self.logger.log_comparison_results(
                comparison_results['comparison_summary'], comparison_type
            )
        
        self.logger.finish_run()
    
    def create_sweep_config(self, sweep_config: Dict) -> str:
        """
        Create a W&B sweep configuration.
        
        Args:
            sweep_config: Sweep configuration dictionary
            
        Returns:
            str: Sweep ID
        """
        if not self.enabled:
            return "sweep_disabled"
        
        sweep_id = wandb.sweep(sweep_config, project=self.logger.project_name)
        logging.info(f"Created W&B sweep: {sweep_id}")
        return sweep_id