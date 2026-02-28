import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import shap
import pandas as pd
from .base_pipeline import BasePipeline

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
        # Call BasePipeline constructor for shared setup
        super().__init__(X_train, X_test, y_train, y_test, args, parameters, feature_names)

        # Torch-specific setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert inputs to tensors
        if isinstance(X_train, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X_train.columns)
            X_train = X_train.to_numpy()
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.to_numpy().ravel()
        if isinstance(y_test, (pd.Series, pd.DataFrame)):
            y_test = y_test.to_numpy().ravel()

        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test  = y_test.astype(np.int64)

        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.X_test  = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        self.y_test  = torch.tensor(y_test, dtype=torch.long).to(self.device)

        # Ensure feature_names is always a Python list
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        else:
            feature_names = list(feature_names)
        self.feature_names = feature_names

        self.selected_feature_names = None

    def initialize_model(self):
        """Initialize MLP with provided parameters."""
        input_dim = self.X_train.shape[1]
        output_dim = len(torch.unique(self.y_train))
        self.model = TorchMLP(
            input_dim=input_dim,
            hidden_layer_sizes=self.parameters.get("hidden_layer_sizes", [100, 50]),
            activation=self.parameters.get("activation", "relu"),
            dropout=self.parameters.get("dropout", 0.3),
            output_dim=output_dim
        ).to(self.device)
        return self

    def fit(self, epochs=50, batch_size=32, lr=1e-3,
            use_feature_selection=False, top_k=10):
        """Train the MLP, optionally apply SHAP-based feature selection."""
        if self.model is None:
            raise ValueError("Model not built. Call initialize_model() first.")

        dataset = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        if use_feature_selection:
            # Compute SHAP-based feature importance
            self.selected_feature_names = self.compute_feature_importance(top_k=top_k)

            # Reduce training and test sets to selected features
            selected_indices = [self.feature_names.index(f) for f in self.selected_feature_names]
            self.X_train = self.X_train[:, selected_indices]
            self.X_test  = self.X_test[:, selected_indices]

            # Reinitialize and retrain model on reduced features
            self.initialize_model()
            dataset = TensorDataset(self.X_train, self.y_train)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            self.model.train()
            for epoch in range(epochs):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    preds = self.model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()
        else:
            self.selected_feature_names = self.feature_names

        return self

    def predict(self, X):
        """Generate predictions for new data."""
        X = self._prepare_input(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor)
            return preds.argmax(dim=1).cpu().numpy()

    def _predict_proba(self, X):
        """Generate class probability estimates for new data."""
        X = self._prepare_input(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor)
            return torch.softmax(preds, dim=1).cpu().numpy()

    def compute_feature_importance(self, top_k=10):
        """Use SHAP to estimate feature importance and select top features."""
        X_background = self.X_train[:100].cpu().numpy()
        X_sample = self.X_test[:50].cpu().numpy()

        def predict_fn(x):
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(x_tensor)
                return torch.softmax(preds, dim=1).cpu().numpy()

        explainer = shap.KernelExplainer(predict_fn, X_background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)

        # Aggregate SHAP values
        if isinstance(shap_values, list):
            per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
            shap_values = np.mean(per_class, axis=0)
        elif shap_values.ndim == 3:
            shap_values = np.abs(shap_values).mean(axis=0).mean(axis=1)
        else:
            shap_values = np.abs(shap_values).mean(axis=0)

        shap_values = np.ravel(shap_values)

        if shap_values.shape[0] != len(self.feature_names):
            raise ValueError(
                f"Mismatch: {shap_values.shape[0]} SHAP values vs {len(self.feature_names)} features."
            )

        indices = np.argsort(shap_values)[::-1][:top_k].tolist()
        selected_feature_names = [self.feature_names[i] for i in indices]

        return selected_feature_names

    def get_selected_features(self):
        """Return the names of features selected by SHAP-based selection.
        If feature selection was not applied, return all original features."""
        if self.selected_feature_names is None:
            return self.feature_names
        return self.selected_feature_names

    def _prepare_input(self, X):
        """Helper to convert input into numpy array of float32."""
        if isinstance(X, pd.DataFrame): 
            X = X.to_numpy().astype(np.float32)
        elif isinstance(X, np.ndarray): 
            X = X.astype(np.float32)
        elif isinstance(X, torch.Tensor): 
            X = X.detach().cpu().numpy().astype(np.float32)
        return X