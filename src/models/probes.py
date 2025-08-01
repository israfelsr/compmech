import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


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

                # # Standardize features
                # scaler = StandardScaler()
                # X_train_scaled = scaler.fit_transform(X_train)
                # X_val_scaled = scaler.transform(X_val)

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


class MLPProbe(nn.Module):
    """Simple MLP probe for more complex attribute prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TorchAttributeProbes:
    """
    PyTorch-based attribute probes for more complex architectures.
    """

    def __init__(
        self,
        dataset=None,
        hidden_dims=[512, 256],
        dropout=0.1,
        learning_rate=0.001,
        batch_size=64,
        epochs=100,
        device="auto",
    ):
        self.dataset = dataset
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def train_single_probe(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        attribute_idx: int,
        cv_folds: int = 5,
    ) -> Optional[Dict]:
        """Train a single MLP probe using PyTorch."""
        y = labels[:, attribute_idx].float()

        # Skip attributes with insufficient examples
        if torch.sum(y) < cv_folds or torch.sum(1 - y) < cv_folds:
            return None

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, val_idx in skf.split(features.numpy(), y.numpy()):
            X_train = features[train_idx].to(self.device)
            X_val = features[val_idx].to(self.device)
            y_train = y[train_idx].to(self.device)
            y_val = y[val_idx].to(self.device)

            # Create and train model
            model = MLPProbe(features.shape[1], self.hidden_dims, 1, self.dropout).to(
                self.device
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.BCEWithLogitsLoss()

            # Training loop
            model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                outputs = model(X_train).squeeze()
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val).squeeze()
                val_preds = torch.sigmoid(val_outputs) > 0.5

                f1 = f1_score(
                    y_val.cpu().numpy(),
                    val_preds.cpu().numpy(),
                    average="binary",
                    zero_division=0,
                )
                fold_scores.append(f1)

        results = {
            "attribute_idx": attribute_idx,
            "mean_f1": np.mean(fold_scores),
            "std_f1": np.std(fold_scores),
            "f1_scores": fold_scores,
            "n_positive": int(torch.sum(y).item()),
            "n_total": len(y),
        }

        if self.dataset and hasattr(self.dataset, "idx_to_attribute"):
            results["attribute_name"] = self.dataset.idx_to_attribute[attribute_idx]

        return results
