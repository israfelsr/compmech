import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datasets import Dataset
from pathlib import Path


def load_layer_features(
    dataset: Dataset,
    model_name: str,
    layer: str,
    features_dir: str = "/home/bzq999/data/compmech/features/",
    prefix: str = "",
):
    if isinstance(layer, list):
        layer = layer[0]

    model_features_dir = Path(features_dir) / model_name
    if not model_features_dir.exists():
        raise FileNotFoundError(
            f"Features directory does not exist: {model_features_dir}"
        )
    feature_dataset_path = model_features_dir / f"{prefix}layer_{layer}.pt"
    logging.info(f"Loading cached features for layer {layer}")
    cached_layers_features = {}
    try:
        cached_layers_features[layer] = torch.load(
            feature_dataset_path, weights_only=False, map_location=torch.device("cpu")
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
            logging.warning(f"No features found for {image_path}, using zero vector")
            feature_dim = len(list(features.values())[0]) if features else 768
            feature_values.append(np.zeros(feature_dim).tolist())

    merged_dataset = merged_dataset.add_column(feature_column_name, feature_values)

    return merged_dataset


class AttributeProbes:
    """
    Reusable class for training and evaluating attribute probes on visual features.
    Supports different probe types and evaluation strategies.
    """

    def __init__(
        self,
        dataset=None,
        layer: str = None,
        probe_type="logistic",
        random_seed=42,
    ):
        """
        Initialize the AttributeProbes class.

        Args:
            dataset: dataset for the examples
            layer: layer as feature
            probe_type: Type of probe to use ('logistic', 'linear', 'mlp')
            random_seed: Random seed for reproducibility
        """
        self.dataset = dataset.to_pandas()
        self.layer = layer
        self.probe_type = probe_type
        self.random_seed = random_seed

        # Pre-compute
        self.features = np.stack(self.dataset[layer].tolist())
        attribute_cols = [
            col
            for col in self.dataset.columns
            if col not in ["image_path", "concept", layer]
        ]
        self.label_arrays = {attr: self.dataset[attr].values for attr in attribute_cols}

        # Create concept to feature indices mapping
        concept_indices = defaultdict(list)
        for i, concept in enumerate(self.dataset["concept"]):
            concept_indices[concept].append(i)

        self.concept_to_indices = {
            concept: np.array(indices, dtype=np.int32)
            for concept, indices in concept_indices.items()
        }

        labels = dataset.remove_columns(["image_path", layer]).to_pandas()
        self.unique_concepts = labels.groupby("concept").first().reset_index()

    def _generate_random_predictions(self, y_true, positive_rate, random_seed=None):
        """Generate random predictions with same class distribution as y_true"""
        np.random.seed(random_seed)

        # Strategy 1: Random predictions with same positive rate as validation set
        y_random = np.random.binomial(1, positive_rate, size=len(y_true))
        return y_random

    def train_single_probe(
        self,
        attribute: str,
        cv_folds: int = 5,
        n_repeats: int = 2,
    ) -> Optional[Dict]:
        """
        Train a single attribute probe using stratified cross-validation.

        Args:
            features: Input features (N, D)
            labels: All attribute labels (N, num_attributes)
            cv_folds: Number of cross-validation folds
            n_repeats: Number of times to repeat CV

        Returns:
            dict: Results containing performance metrics
        """
        # Prepare data
        concepts = self.unique_concepts["concept"]
        labels = self.unique_concepts[attribute]
        y = self.label_arrays[attribute]

        # Skip attributes with insufficient positive examples
        if np.sum(labels) < cv_folds or np.sum(1 - labels) < cv_folds:
            logging.warning(f"Skipping attribute {attribute}: insufficient examples")
            return None

        all_scores = {"f1": [], "accuracy": [], "precision": [], "recall": []}
        baseline_scores = {"f1": [], "accuracy": [], "precision": [], "recall": []}

        # Repeat cross-validation multiple times
        for repeat in range(n_repeats):
            skf = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=self.random_seed + repeat
            )

            for train_idx, val_idx in skf.split(concepts, labels):
                concepts_train = concepts.iloc[train_idx]
                concepts_val = concepts.iloc[val_idx]
                train_mask = np.concatenate(
                    [self.concept_to_indices[concept] for concept in concepts_train]
                )
                val_mask = np.concatenate(
                    [self.concept_to_indices[concept] for concept in concepts_val]
                )

                X_train = self.features[train_mask]
                X_val = self.features[val_mask]
                y_train = y[train_mask]
                y_val = y[val_mask]

                # Train probe
                probe = self._create_probe()
                probe.fit(X_train, y_train)

                # Make predictions
                y_pred = probe.predict(X_val)

                # Calculate metrics
                all_scores["f1"].append(
                    f1_score(y_val, y_pred, average="binary", zero_division=0)
                )
                all_scores["accuracy"].append(accuracy_score(y_val, y_pred))
                all_scores["precision"].append(
                    precision_score(y_val, y_pred, average="binary", zero_division=0)
                )
                all_scores["recall"].append(
                    recall_score(y_val, y_pred, average="binary", zero_division=0)
                )

                # Random baseline evaluation
                y_random = self._generate_random_predictions(
                    y_val,
                    np.mean(labels.iloc[train_idx]),
                    random_seed=self.random_seed + repeat,
                )

                # Calculate baseline metrics
                baseline_scores["f1"].append(
                    f1_score(y_val, y_random, average="binary", zero_division=0)
                )
                baseline_scores["accuracy"].append(accuracy_score(y_val, y_random))
                baseline_scores["precision"].append(
                    precision_score(y_val, y_random, average="binary", zero_division=0)
                )
                baseline_scores["recall"].append(
                    recall_score(y_val, y_random, average="binary", zero_division=0)
                )

        # Create results dictionary
        results = {
            "attribute": attribute,
            "n_positive": int(np.sum(y)),
            "n_total": len(y),
        }

        # Add mean and std for each metric
        for metric, scores in all_scores.items():
            results[f"mean_{metric}"] = np.mean(scores)
            results[f"std_{metric}"] = np.std(scores)
            results[f"{metric}_scores"] = scores

        # Add baseline results
        for metric, scores in baseline_scores.items():
            results[f"baseline_mean_{metric}"] = np.mean(scores)
            results[f"baseline_std_{metric}"] = np.std(scores)
            results[f"baseline_{metric}_scores"] = scores

        # Add improvement over baseline
        for metric in ["f1", "accuracy", "precision", "recall"]:
            improvement = results[f"mean_{metric}"] - results[f"baseline_mean_{metric}"]
            results[f"{metric}_improvement"] = improvement

        return results

    def train_all_probes(
        self,
        cv_folds: int = 5,
        n_repeats: int = 2,
    ) -> Dict:
        """
        Train probes for all attributes.

        Args:
            features: Input features (N, D)
            labels: All attribute labels (N, num_attributes)
            cv_folds: Number of cross-validation folds
            n_repeats: Number of times to repeat CV

        Returns:
            dict: Complete results for all attributes
        """
        logging.info(f"Training {self.probe_type} probes for all attributes...")

        all_results = []
        column_names = list(self.label_arrays.keys())

        for attr in tqdm(column_names, desc="Training attribute probes"):
            results = self.train_single_probe(attr, cv_folds, n_repeats)
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
            "probe_type": self.probe_type,
            "cv_folds": cv_folds,
            "n_repeats": n_repeats,
        }

    def evaluate_specific_attributes(
        self,
        attributes: List[str],
        cv_folds: int = 5,
        n_repeats: int = 2,
    ) -> Dict:
        """
        Train probes for specific attributes only.

        Args:
            attributes: List of attribute indices to evaluate
            cv_folds: Number of cross-validation folds
            n_repeats: Number of times to repeat CV

        Returns:
            dict: Results for specified attributes
        """
        logging.info(f"Training probes for {len(attributes)} specific attributes...")

        all_results = []
        for attr in tqdm(attributes, desc="Training specific attribute probes"):
            results = self.train_single_probe(attr, cv_folds, n_repeats)
            if results is not None:
                all_results.append(results)
                logging.info(
                    f"Attribute {attr}: F1 = {results['mean_f1']:.4f} ± {results['std_f1']:.4f}"
                )

        summary = self._calculate_summary(all_results)

        return {
            "individual_results": all_results,
            "summary": summary,
            "probe_type": self.probe_type,
            "cv_folds": cv_folds,
            "n_repeats": n_repeats,
            "tested_attributes": attributes,
        }

    def _create_probe(self):
        """Create a probe based on the specified type."""
        if self.probe_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_seed,
                n_jobs=-1,
                tol=1e-3,
            )
        elif self.probe_type == "linear":
            from sklearn.linear_model import LinearRegression

            return LinearRegression()
        elif self.probe_type == "mlp":
            from sklearn.neural_network import MLPClassifier

            return MLPClassifier(
                hidden_layer_sizes=(100,), max_iter=1000, random_state=self.random_seed
            )
        else:
            raise ValueError(f"Unknown probe type: {self.probe_type}")

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


class SpatialProbes:
    def __init__(
        self,
        dataset=None,
        layer: str = None,
        probe_type="logistic",
        random_seed=42,
        filter_relations=None,
    ):
        """
        Initialize the SpatialProbes class.

        Args:
            dataset: dataset with 'spatial_relation' column
            layer: layer name to use as features
            probe_type: Type of probe to use ('logistic', 'linear', 'mlp')
            random_seed: Random seed for reproducibility
            filter_relations: List of relations to keep (e.g., ['top', 'right', 'left', 'bottom'])
                            If None, keeps all relations except 'behind'
        """
        self.layer = layer
        self.probe_type = probe_type
        self.random_seed = random_seed

        # Filter dataset
        if filter_relations is None:
            # Default: remove 'behind'
            filter_relations = ["top", "right", "left", "bottom"]

        dataset = dataset.filter(lambda x: x["spatial_relation"] in filter_relations)
        logging.info(
            f"Filtered dataset to {len(dataset)} samples with relations: {filter_relations}"
        )

        self.dataset = dataset.to_pandas()
        self.features = np.stack(self.dataset[layer].tolist())
        self.labels = self.dataset["spatial_relation"].values
        self.class_names = sorted(list(set(self.labels)))
        self.n_classes = len(self.class_names)

        logging.info(f"Classes: {self.class_names}")
        logging.info(
            f"Class distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}"
        )

    def _create_probe(self):
        """Create a probe based on the specified type."""
        if self.probe_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_seed,
                n_jobs=-1,
                multi_class="multinomial",
                solver="lbfgs",
            )
        elif self.probe_type == "linear":
            from sklearn.svm import LinearSVC

            return LinearSVC(max_iter=1000, random_state=self.random_seed)
        elif self.probe_type == "mlp":
            from sklearn.neural_network import MLPClassifier

            return MLPClassifier(
                hidden_layer_sizes=(100,), max_iter=1000, random_state=self.random_seed
            )
        else:
            raise ValueError(f"Unknown probe type: {self.probe_type}")

    def _generate_random_predictions(
        self, y_true, class_distribution, random_seed=None
    ):
        """Generate random predictions based on class distribution"""
        np.random.seed(random_seed)
        classes = list(class_distribution.keys())
        probs = list(class_distribution.values())
        probs = np.array(probs) / sum(probs)
        y_random = np.random.choice(classes, size=len(y_true), p=probs)
        return y_random

    def train_probe(
        self,
        cv_folds: int = 5,
        n_repeats: int = 2,
    ) -> Dict:
        """
        Train spatial relation probe using stratified cross-validation.

        Args:
            cv_folds: Number of cross-validation folds
            n_repeats: Number of times to repeat CV

        Returns:
            dict: Results containing performance metrics
        """
        X = self.features
        y = self.labels

        all_scores = {
            "f1_macro": [],
            "f1_weighted": [],
            "accuracy": [],
            "precision_macro": [],
            "precision_weighted": [],
            "recall_macro": [],
            "recall_weighted": [],
        }
        baseline_scores = {
            "f1_macro": [],
            "f1_weighted": [],
            "accuracy": [],
            "precision_macro": [],
            "precision_weighted": [],
            "recall_macro": [],
            "recall_weighted": [],
        }

        all_confusion_matrices = []

        # Repeat cross-validation multiple times
        for repeat in range(n_repeats):
            skf = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=self.random_seed + repeat
            )

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train probe
                probe = self._create_probe()
                probe.fit(X_train, y_train)

                # Make predictions
                y_pred = probe.predict(X_val)

                # Calculate metrics
                all_scores["f1_macro"].append(
                    f1_score(y_val, y_pred, average="macro", zero_division=0)
                )
                all_scores["f1_weighted"].append(
                    f1_score(y_val, y_pred, average="weighted", zero_division=0)
                )
                all_scores["accuracy"].append(accuracy_score(y_val, y_pred))
                all_scores["precision_macro"].append(
                    precision_score(y_val, y_pred, average="macro", zero_division=0)
                )
                all_scores["precision_weighted"].append(
                    precision_score(y_val, y_pred, average="weighted", zero_division=0)
                )
                all_scores["recall_macro"].append(
                    recall_score(y_val, y_pred, average="macro", zero_division=0)
                )
                all_scores["recall_weighted"].append(
                    recall_score(y_val, y_pred, average="weighted", zero_division=0)
                )

                # Confusion matrix
                cm = confusion_matrix(y_val, y_pred, labels=self.class_names)
                all_confusion_matrices.append(cm)

                # Random baseline evaluation
                train_class_dist = dict(zip(*np.unique(y_train, return_counts=True)))
                y_random = self._generate_random_predictions(
                    y_val, train_class_dist, random_seed=self.random_seed + repeat
                )

                # Calculate baseline metrics
                baseline_scores["f1_macro"].append(
                    f1_score(y_val, y_random, average="macro", zero_division=0)
                )
                baseline_scores["f1_weighted"].append(
                    f1_score(y_val, y_random, average="weighted", zero_division=0)
                )
                baseline_scores["accuracy"].append(accuracy_score(y_val, y_random))
                baseline_scores["precision_macro"].append(
                    precision_score(y_val, y_random, average="macro", zero_division=0)
                )
                baseline_scores["precision_weighted"].append(
                    precision_score(
                        y_val, y_random, average="weighted", zero_division=0
                    )
                )
                baseline_scores["recall_macro"].append(
                    recall_score(y_val, y_random, average="macro", zero_division=0)
                )
                baseline_scores["recall_weighted"].append(
                    recall_score(y_val, y_random, average="weighted", zero_division=0)
                )

        # Average confusion matrix
        avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0)

        # Create results dictionary
        results = {
            "task": "spatial_relation_classification",
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "n_samples": len(y),
            "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
        }

        # Add mean and std for each metric
        for metric, scores in all_scores.items():
            results[f"mean_{metric}"] = np.mean(scores)
            results[f"std_{metric}"] = np.std(scores)
            results[f"{metric}_scores"] = scores

        # Add baseline results
        for metric, scores in baseline_scores.items():
            results[f"baseline_mean_{metric}"] = np.mean(scores)
            results[f"baseline_std_{metric}"] = np.std(scores)
            results[f"baseline_{metric}_scores"] = scores

        # Add improvement over baseline
        for metric in all_scores.keys():
            improvement = results[f"mean_{metric}"] - results[f"baseline_mean_{metric}"]
            results[f"{metric}_improvement"] = improvement

        # Add confusion matrix
        results["avg_confusion_matrix"] = avg_confusion_matrix.tolist()
        results["confusion_matrix_labels"] = self.class_names

        return results
