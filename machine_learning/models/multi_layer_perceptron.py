import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import shap

from .base_pipeline import BasePipeline
from machine_learning.utils.training.data_evaluation import data_evaluation


class TorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=[100, 50],
                 activation="relu", dropout=0.3, output_dim=None):
        super().__init__()
        layers = []
        prev_dim = input_dim
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiLayerPerceptronPipeline(BasePipeline):
    def __init__(self, X_train, X_test, y_train, y_test, args, parameters=None, feature_names=None):
        super().__init__(X_train, X_test, y_train, y_test, args, parameters, feature_names)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(X_train, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X_train.columns)
            X_train = X_train.to_numpy()
        if isinstance(X_test,  pd.DataFrame):
            X_test  = X_test.to_numpy()
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.to_numpy().ravel()
        if isinstance(y_test,  (pd.Series, pd.DataFrame)):
            y_test  = y_test.to_numpy().ravel()

        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test  = y_test.astype(np.int64)

        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.X_test  = torch.tensor(X_test,  dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        self.y_test  = torch.tensor(y_test,  dtype=torch.long).to(self.device)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        self.original_feature_names = list(feature_names)

    def initialize_model(self):
        """Initialize MLP with provided parameters."""
        input_dim  = self.X_train.shape[1]
        output_dim = len(torch.unique(self.y_train))
        self.model = TorchMLP(
            input_dim=input_dim,
            hidden_layer_sizes=self.parameters.get("hidden_layer_sizes", [100, 50]),
            activation=self.parameters.get("activation", "relu"),
            dropout=self.parameters.get("dropout", 0.3),
            output_dim=output_dim,
        ).to(self.device)
        return self

    def fit(self, epochs=50, batch_size=32, lr=1e-3,
            use_feature_selection=False, top_k=15):
        """Train the MLP, optionally applying SHAP-based feature selection."""
        if self.model is None:
            raise ValueError("Model not built. Call initialize_model() first.")

        self._run_training_loop(epochs, batch_size, lr)

        if use_feature_selection:
            shap_scores, selected_names = self._compute_shap_importance(top_k=top_k)

            selected_indices = [self.original_feature_names.index(f) for f in selected_names]

            # Build boolean mask so _set_features stays consistent with other pipelines
            mask = np.zeros(len(self.original_feature_names), dtype=bool)
            mask[selected_indices] = True

            self._set_features(
                method=f"SHAP KernelExplainer (top_k={top_k})",
                mask=mask,
                importances=shap_scores,
            )

            self.X_train = self.X_train[:, selected_indices]
            self.X_test  = self.X_test[:, selected_indices]

            self.initialize_model()
            self._run_training_loop(epochs, batch_size, lr)
        else:
            self._set_features(method="none")

        return self

    def _run_training_loop(self, epochs, batch_size, lr):
        """Run the standard Adam + CrossEntropy training loop."""
        dataset   = TensorDataset(self.X_train, self.y_train)
        loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

    def _compute_shap_importance(self, top_k=10):
        """
        Use SHAP KernelExplainer to compute per-feature importance scores.

        Returns
        -------
        shap_scores : np.ndarray of shape (n_features,)
            Mean absolute SHAP value for every original feature.
        selected_names : list of str
            Names of the top_k features ranked by descending importance.
        """
        X_background = self.X_train[:100].cpu().numpy()
        X_sample     = self.X_test[:50].cpu().numpy()

        def predict_fn(x):
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            self.model.eval()
            with torch.no_grad():
                return torch.softmax(self.model(x_tensor), dim=1).cpu().numpy()

        explainer   = shap.KernelExplainer(predict_fn, X_background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)

        if isinstance(shap_values, list):
            shap_scores = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif shap_values.ndim == 3:
            shap_scores = np.abs(shap_values).mean(axis=0).mean(axis=1)
        else:
            shap_scores = np.abs(shap_values).mean(axis=0)

        shap_scores = np.ravel(shap_scores)

        if shap_scores.shape[0] != len(self.original_feature_names):
            raise ValueError(
                f"Mismatch: {shap_scores.shape[0]} SHAP values vs "
                f"{len(self.original_feature_names)} features."
            )

        top_indices    = np.argsort(shap_scores)[::-1][:top_k].tolist()
        selected_names = [self.original_feature_names[i] for i in top_indices]

        return shap_scores, selected_names

    def predict(self, X):
        """Generate predictions for new data."""
        X_tensor = torch.tensor(self._prepare_input(X), dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor).argmax(dim=1).cpu().numpy()

    def _predict_proba(self, X):
        """Generate class probability estimates for new data."""
        X_tensor = torch.tensor(self._prepare_input(X), dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(X_tensor), dim=1).cpu().numpy()

    def evaluate(self):
        """Evaluate the model using data_evaluation."""
        y_pred  = self.predict(self.X_test)
        y_proba = self._predict_proba(self.X_test)
        y_true  = self.y_test.cpu().numpy() if isinstance(self.y_test, torch.Tensor) else self.y_test
        self.results = data_evaluation(y_pred, y_true, y_proba=y_proba)
        return self

    def _prepare_input(self, X):
        """Convert input to a float32 NumPy array regardless of source type."""
        if isinstance(X, pd.DataFrame):
            return X.to_numpy().astype(np.float32)
        if isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy().astype(np.float32)
        return np.array(X, dtype=np.float32)