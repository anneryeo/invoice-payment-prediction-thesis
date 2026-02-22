import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from ..Utils.data_evaluation import data_evaluation

class TorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=[100, 50], 
                 activation="relu", dropout=0.3, output_dim=None):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Map activation string to function
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh
        
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MultiLayerPerceptronPipeline:
    def __init__(self, X_train, X_test, y_train, y_test, args, parameters=None):
        self.args = args
        self.parameters = parameters or {}
        self.model = None
        self.results = None

        # Convert DataFrames/Series to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.to_numpy().ravel()
        if isinstance(y_test, (pd.Series, pd.DataFrame)):
            y_test = y_test.to_numpy().ravel()

        # Ensure numeric dtype
        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test  = y_test.astype(np.int64)

        # Convert to torch tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_test  = torch.tensor(X_test, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_test  = torch.tensor(y_test, dtype=torch.long)

    def build_model(self):
        input_dim = self.X_train.shape[1]
        output_dim = len(torch.unique(self.y_train))

        self.model = TorchMLP(
            input_dim=input_dim,
            hidden_layer_sizes=self.parameters.get("hidden_layer_sizes", [100, 50]),
            activation=self.parameters.get("activation", "relu"),
            dropout=self.parameters.get("dropout", 0.3),
            output_dim=output_dim
        )
        return self

    def train(self, epochs=50, batch_size=32, lr=1e-3):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        dataset = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        """
        Generate predictions for new data.
        Accepts either numpy arrays or torch tensors.
        Returns numpy array of predicted class labels.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy().astype(np.float32)
        elif isinstance(X, np.ndarray):
            X = X.astype(np.float32)
        elif isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy().astype(np.float32)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor)
            predicted_classes = preds.argmax(dim=1).cpu().numpy()
        return predicted_classes

    def evaluation(self):
        """
        Evaluate the model using the simplified data_evaluation
        that expects (y_pred, y_test).
        """
        y_pred = self.predict(self.X_test)
        self.results = data_evaluation(y_pred, self.y_test.numpy())
        return self

    def show_results(self):
        return self.results