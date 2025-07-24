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


class AttributeProbes:
    """
    Reusable class for training and evaluating attribute probes on visual features.
    Supports different probe types and evaluation strategies.
    """

    def __init__(self, dataset=None, probe_type="logistic", random_seed=42):
        """
        Initialize the AttributeProbes class.

        Args:
            dataset: Dataset containing attribute information
            probe_type: Type of probe to use ('logistic', 'linear', 'mlp')
            random_seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.probe_type = probe_type
        self.random_seed = random_seed
        self.trained_probes = {}

    def train_single_probe(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        attribute_idx: int,
        cv_folds: int = 5,
        n_repeats: int = 2,
    ) -> Optional[Dict]:
        """
        Train a single attribute probe using stratified cross-validation.

        Args:
            features: Input features (N, D)
            labels: All attribute labels (N, num_attributes)
            attribute_idx: Index of the target attribute
            cv_folds: Number of cross-validation folds
            n_repeats: Number of times to repeat CV

        Returns:
            dict: Results containing performance metrics
        """
        # Get binary labels for this specific attribute
        y = labels[:, attribute_idx]

        # Skip attributes with insufficient positive examples
        if np.sum(y) < cv_folds or np.sum(1 - y) < cv_folds:
            logging.warning(
                f"Skipping attribute {attribute_idx}: insufficient examples"
            )
            return None

        all_scores = {"f1": [], "accuracy": [], "precision": [], "recall": []}

        # Repeat cross-validation multiple times
        for repeat in range(n_repeats):
            skf = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=self.random_seed + repeat
            )

            for train_idx, val_idx in skf.split(features, y):
                X_train, X_val = features[train_idx], features[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

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

        # Create results dictionary
        results = {
            "attribute_idx": attribute_idx,
            "n_positive": int(np.sum(y)),
            "n_total": len(y),
        }

        # Add dataset attribute name if available
        if self.dataset and hasattr(self.dataset, "idx_to_attribute"):
            results["attribute_name"] = self.dataset.idx_to_attribute[attribute_idx]

        # Add mean and std for each metric
        for metric, scores in all_scores.items():
            results[f"mean_{metric}"] = np.mean(scores)
            results[f"std_{metric}"] = np.std(scores)
            results[f"{metric}_scores"] = scores

        return results

    def train_all_probes(
        self,
        features: np.ndarray,
        labels: np.ndarray,
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
        num_attributes = labels.shape[1]

        for attr_idx in tqdm(range(num_attributes), desc="Training attribute probes"):
            results = self.train_single_probe(
                features, labels, attr_idx, cv_folds, n_repeats
            )
            if results is not None:
                all_results.append(results)
                logging.info(
                    f"Attribute {attr_idx}: F1 = {results['mean_f1']:.4f} ± {results['std_f1']:.4f}"
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
        features: np.ndarray,
        labels: np.ndarray,
        attribute_indices: List[int],
        cv_folds: int = 5,
        n_repeats: int = 2,
    ) -> Dict:
        """
        Train probes for specific attributes only.

        Args:
            features: Input features (N, D)
            labels: All attribute labels (N, num_attributes)
            attribute_indices: List of attribute indices to evaluate
            cv_folds: Number of cross-validation folds
            n_repeats: Number of times to repeat CV

        Returns:
            dict: Results for specified attributes
        """
        logging.info(
            f"Training probes for {len(attribute_indices)} specific attributes..."
        )

        all_results = []

        for attr_idx in tqdm(
            attribute_indices, desc="Training specific attribute probes"
        ):
            results = self.train_single_probe(
                features, labels, attr_idx, cv_folds, n_repeats
            )
            if results is not None:
                all_results.append(results)
                logging.info(
                    f"Attribute {attr_idx}: F1 = {results['mean_f1']:.4f} ± {results['std_f1']:.4f}"
                )

        summary = self._calculate_summary(all_results)

        return {
            "individual_results": all_results,
            "summary": summary,
            "probe_type": self.probe_type,
            "cv_folds": cv_folds,
            "n_repeats": n_repeats,
            "tested_attributes": attribute_indices,
        }

    def _create_probe(self):
        """Create a probe based on the specified type."""
        if self.probe_type == "logistic":
            return LogisticRegression(max_iter=1000, random_state=self.random_seed)
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
