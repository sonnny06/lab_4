import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader


class MLPRegressor(nn.Module):
    """Same MLP architecture used in the notebook: input -> 32 -> 16 -> 1."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.model(x)


def load_dataset():
    data = fetch_california_housing()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    df["Target"] = y
    return X, y, feature_names, df


def prepare_data(test_size: float = 0.2, random_state: int = 42):
    X, y, feature_names, df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }


def train_mlp(epochs: int = 100, batch_size: int = 32, lr: float = 0.001, seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    bundle = prepare_data(random_state=seed)
    X_train_scaled = bundle["X_train_scaled"]
    X_test_scaled = bundle["X_test_scaled"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train_scaled.shape[1]
    model = MLPRegressor(input_dim)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    y_pred_mlp = y_pred_tensor.numpy().flatten()

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred_mlp),
        "MAE": mean_absolute_error(y_test, y_pred_mlp),
        "R2": r2_score(y_test, y_pred_mlp),
    }

    bundle.update({
        "model": model,
        "train_losses": train_losses,
        "y_pred_mlp": y_pred_mlp,
        "metrics": metrics,
    })
    return bundle


def predict_one(model, scaler, input_values):
    """input_values must follow California Housing feature order."""
    x = np.array(input_values, dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor).item()
    return pred
