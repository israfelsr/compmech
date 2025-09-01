#!/usr/bin/env python3
"""
Linear probe training script using HuggingFace Trainer and PyTorch.
Trains a simple linear layer (no activation) for binary attribute classification.
"""

import yaml
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datasets import Dataset, load_from_disk
import sys
import os
import json
from typing import List, Dict, Any, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import defaultdict

# HuggingFace imports
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset as TorchDataset

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.models.probes import load_layer_features


class LinearProbe(nn.Module):
    """
    Simple linear probe: Linear layer with no activation function.
    Input: feature_dim -> Output: 1 (binary classification)
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        # Initialize with small weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, features):
        return self.linear(features).squeeze(-1)


class AttributeDataset(TorchDataset):
    """Custom dataset for attribute classification."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }


class LinearProbeTrainer:
    """Trainer for linear probes using HuggingFace Trainer."""
    
    def __init__(
        self,
        dataset: Dataset,
        layer: str,
        random_seed: int = 42,
        output_dir: str = "results/linear_probes",
    ):
        self.dataset = dataset.to_pandas()
        self.layer = layer
        self.random_seed = random_seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract features and prepare data
        self.features = np.stack(self.dataset[layer].tolist())
        self.input_dim = self.features.shape[1]
        
        # Get attribute columns (excluding metadata)
        attribute_cols = [
            col for col in self.dataset.columns 
            if col not in ["image_path", "concept", layer]
        ]
        self.label_arrays = {attr: self.dataset[attr].values for attr in attribute_cols}
        
        # Create concept to feature indices mapping for cross-validation
        concept_indices = defaultdict(list)
        for i, concept in enumerate(self.dataset["concept"]):
            concept_indices[concept].append(i)
        
        self.concept_to_indices = {
            concept: np.array(indices, dtype=np.int32)
            for concept, indices in concept_indices.items()
        }
        
        # Get unique concepts for stratified CV
        labels_df = Dataset.from_pandas(self.dataset).remove_columns(["image_path", layer]).to_pandas()
        self.unique_concepts = labels_df.groupby("concept").first().reset_index()
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        # Apply sigmoid to get probabilities, then threshold at 0.5
        probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        preds = (probs > 0.5).astype(int)
        labels = labels.astype(int)
        
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="binary", zero_division=0),
            "precision": precision_score(labels, preds, average="binary", zero_division=0),
            "recall": recall_score(labels, preds, average="binary", zero_division=0),
        }
    
    def train_single_probe(
        self,
        attribute: str,
        cv_folds: int = 5,
        n_repeats: int = 2,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 0.01,
    ) -> Optional[Dict]:
        """Train a single linear probe using cross-validation."""
        
        # Prepare data
        concepts = self.unique_concepts["concept"]
        labels = self.unique_concepts[attribute]
        y = self.label_arrays[attribute]
        
        # Skip attributes with insufficient positive examples
        if np.sum(labels) < cv_folds or np.sum(1 - labels) < cv_folds:
            logging.warning(f"Skipping attribute {attribute}: insufficient examples")
            return None
        
        all_scores = {"f1": [], "accuracy": [], "precision": [], "recall": []}
        
        # Repeat cross-validation multiple times
        for repeat in range(n_repeats):
            skf = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=self.random_seed + repeat
            )
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(concepts, labels)):
                # Get training and validation concepts
                concepts_train = concepts.iloc[train_idx]
                concepts_val = concepts.iloc[val_idx]
                
                # Map concepts to sample indices
                train_mask = np.concatenate([
                    self.concept_to_indices[concept] for concept in concepts_train
                ])
                val_mask = np.concatenate([
                    self.concept_to_indices[concept] for concept in concepts_val
                ])
                
                # Prepare datasets
                train_features = self.features[train_mask]
                val_features = self.features[val_mask]
                train_labels = y[train_mask].astype(np.float32)
                val_labels = y[val_mask].astype(np.float32)
                
                train_dataset = AttributeDataset(train_features, train_labels)
                val_dataset = AttributeDataset(val_features, val_labels)
                
                # Initialize model
                model = LinearProbe(self.input_dim)
                
                # Training arguments
                training_args = TrainingArguments(
                    output_dir=self.output_dir / f"tmp_{attribute}_r{repeat}_f{fold}",
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    logging_steps=10,
                    eval_steps=50,
                    evaluation_strategy="steps",
                    save_strategy="no",  # Don't save checkpoints
                    load_best_model_at_end=True,
                    metric_for_best_model="f1",
                    greater_is_better=True,
                    seed=self.random_seed,
                    remove_unused_columns=False,
                    dataloader_drop_last=False,
                    report_to=None,  # Disable wandb/tensorboard
                )
                
                # Initialize trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=self.compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                )
                
                # Train the model
                trainer.train()
                
                # Evaluate
                eval_results = trainer.evaluate()
                
                # Store results
                all_scores["f1"].append(eval_results["eval_f1"])
                all_scores["accuracy"].append(eval_results["eval_accuracy"])
                all_scores["precision"].append(eval_results["eval_precision"])
                all_scores["recall"].append(eval_results["eval_recall"])
                
                # Clean up temporary directory
                import shutil
                temp_dir = self.output_dir / f"tmp_{attribute}_r{repeat}_f{fold}"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        
        # Create results dictionary
        results = {
            "attribute": attribute,
            "n_positive": int(np.sum(y)),
            "n_total": len(y),
            "input_dim": self.input_dim,
        }
        
        # Add mean and std for each metric
        for metric, scores in all_scores.items():
            results[f"mean_{metric}"] = np.mean(scores)
            results[f"std_{metric}"] = np.std(scores)
            results[f"{metric}_scores"] = scores
        
        return results
    
    def train_all_probes(
        self,
        cv_folds: int = 5,
        n_repeats: int = 2,
        **training_kwargs
    ) -> Dict:
        """Train linear probes for all attributes."""
        logging.info(f"Training linear probes for all attributes...")
        
        all_results = []
        attribute_names = list(self.label_arrays.keys())
        
        for attr in attribute_names:
            logging.info(f"Training probe for attribute: {attr}")
            results = self.train_single_probe(attr, cv_folds, n_repeats, **training_kwargs)
            if results is not None:
                all_results.append(results)
                logging.info(
                    f"Attribute {attr}: F1 = {results['mean_f1']:.4f} ± {results['std_f1']:.4f}"
                )
        
        # Calculate summary statistics
        summary = self._calculate_summary(all_results)
        
        return {
            "individual_results": all_results,
            "summary": summary,
            "probe_type": "linear_torch",
            "layer": self.layer,
            "cv_folds": cv_folds,
            "n_repeats": n_repeats,
        }
    
    def train_specific_probes(
        self,
        attributes: List[str],
        cv_folds: int = 5,
        n_repeats: int = 2,
        **training_kwargs
    ) -> Dict:
        """Train linear probes for specific attributes."""
        logging.info(f"Training linear probes for {len(attributes)} specific attributes...")
        
        all_results = []
        for attr in attributes:
            logging.info(f"Training probe for attribute: {attr}")
            results = self.train_single_probe(attr, cv_folds, n_repeats, **training_kwargs)
            if results is not None:
                all_results.append(results)
                logging.info(
                    f"Attribute {attr}: F1 = {results['mean_f1']:.4f} ± {results['std_f1']:.4f}"
                )
        
        summary = self._calculate_summary(all_results)
        
        return {
            "individual_results": all_results,
            "summary": summary,
            "probe_type": "linear_torch",
            "layer": self.layer,
            "cv_folds": cv_folds,
            "n_repeats": n_repeats,
            "tested_attributes": attributes,
        }
    
    def _calculate_summary(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics across all results."""
        if not results:
            return {}
        
        metrics = ["f1", "accuracy", "precision", "recall"]
        summary = {"n_attributes_tested": len(results)}
        
        for metric in metrics:
            scores = [r[f"mean_{metric}"] for r in results]
            summary[f"mean_{metric}_across_attributes"] = np.mean(scores)
            summary[f"std_{metric}_across_attributes"] = np.std(scores)
            summary[f"median_{metric}_across_attributes"] = np.median(scores)
            summary[f"min_{metric}"] = np.min(scores)
            summary[f"max_{metric}"] = np.max(scores)
        
        return summary


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("linear_probe_training.log"),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train linear probes using HuggingFace Trainer")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--dataset_path", type=str, help="Override dataset path from config"
    )
    parser.add_argument("--layer", type=str, help="Override layer")
    parser.add_argument("--cv-folds", type=int, help="Override cross-validation folds")
    parser.add_argument("--n-repeats", type=int, help="Override number of CV repeats")
    parser.add_argument("--output_dir", type=str, help="Override directory for results")
    parser.add_argument(
        "--specific-attributes",
        nargs="+",
        type=str,
        help="Train probes only for specific attributes",
    )
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    logging.info(f"Loaded configuration from {args.config}")
    
    # Override paths if provided
    dataset_path = args.dataset_path or config["dataset"]["path"]
    dataset = load_from_disk(dataset_path)
    
    model_config = config["model"]
    if args.layer:
        layer_idx = args.layer
    else:
        # Try to get layer from config
        layer_idx = model_config.get("layers", [0])
        if isinstance(layer_idx, list):
            layer_idx = layer_idx[0]
    
    logging.info(f"Will probe layer: {layer_idx}")
    
    # Load layer features
    dataset = load_layer_features(
        dataset=dataset,
        model_name=model_config["model_name"],
        layer=layer_idx,
        features_dir=model_config["features_dir"],
    )
    
    # Initialize linear probe trainer
    output_dir = args.output_dir or config["probe"].get("output_dir", "results/linear_probes")
    trainer = LinearProbeTrainer(
        dataset=dataset,
        layer=f"lang_layer_{layer_idx}",
        random_seed=config["probe"]["seed"],
        output_dir=output_dir,
    )
    
    # Training parameters
    training_kwargs = {
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
    }
    
    probe_config = config["probe"]
    cv_folds = args.cv_folds or probe_config["cv_folds"]
    n_repeats = args.n_repeats or probe_config["n_repeats"]
    
    # Train probes
    if args.specific_attributes:
        logging.info(f"Training probes for specific attributes: {args.specific_attributes}")
        results = trainer.train_specific_probes(
            attributes=args.specific_attributes,
            cv_folds=cv_folds,
            n_repeats=n_repeats,
            **training_kwargs
        )
    elif probe_config.get("specific_attribute"):
        logging.info(f"Training probes for specific attributes: {probe_config['specific_attribute']}")
        results = trainer.train_specific_probes(
            attributes=probe_config["specific_attribute"],
            cv_folds=cv_folds,
            n_repeats=n_repeats,
            **training_kwargs
        )
    else:
        logging.info("Training probes for all attributes")
        results = trainer.train_all_probes(
            cv_folds=cv_folds,
            n_repeats=n_repeats,
            **training_kwargs
        )
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results filename
    results_filename = f"linear_probe_results_{layer_idx}.json"
    results_path = output_dir / results_filename
    
    # Save results to JSON
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Results saved to {results_path}")
    
    # Print summary
    print(f"\nLinear Probe Training Summary:")
    print(f"- Layer used: {layer_idx}")
    print(f"- Feature dimension: {trainer.input_dim}")
    print(f"- Number of samples: {trainer.features.shape[0]}")
    print(f"- Number of attributes tested: {results['summary']['n_attributes_tested']}")
    print(f"- Mean F1 score: {results['summary']['mean_f1_across_attributes']:.4f} ± {results['summary']['std_f1_across_attributes']:.4f}")
    print(f"- Mean accuracy: {results['summary']['mean_accuracy_across_attributes']:.4f} ± {results['summary']['std_accuracy_across_attributes']:.4f}")
    print(f"- Results saved to: {results_path}")


if __name__ == "__main__":
    main()