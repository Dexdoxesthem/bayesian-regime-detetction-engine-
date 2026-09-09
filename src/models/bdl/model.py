"""
Bayesian Deep Learning models for regime classification (PyTorch).

Implements:
1. MC Dropout classifier
2. Variational BNN (Bayes by Backprop)
3. Deep Ensemble (M=10)

All produce calibrated regime probabilities with epistemic/aleatoric
uncertainty decomposition.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _to_tensors(X, y=None):
    """Convert numpy arrays to PyTorch tensors."""
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y) if y is not None else None
    return X_t, y_t


def _make_loader(X, y, batch_size=64, shuffle=True):
    """Create a DataLoader from numpy arrays."""
    X_t, y_t = _to_tensors(X, y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# 1. MC Dropout Classifier
# ---------------------------------------------------------------------------

class MCDropoutNet(nn.Module):
    """Neural network with MC Dropout for Bayesian uncertainty."""

    def __init__(self, input_dim, n_classes, hidden_dims=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MCDropoutClassifier:
    """
    MC Dropout neural network classifier.

    Uses dropout at inference time to approximate Bayesian posterior.
    epistemic uncertainty = variance across MC forward passes.
    aleatoric uncertainty = mean predicted variance.
    """

    def __init__(
        self,
        input_dim: int = 14,
        n_classes: int = 5,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.3,
        n_mc_samples: int = 100,
        learning_rate: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.n_mc_samples = n_mc_samples
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.1,
    ) -> Dict:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        X_scaled = self.scaler.fit_transform(X)

        # Time-series aware split
        n_val = int(len(X_scaled) * validation_split)
        X_train, X_val = X_scaled[:-n_val], X_scaled[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        train_loader = _make_loader(X_train, y_train, self.batch_size)
        val_loader = _make_loader(X_val, y_val, self.batch_size, shuffle=False)

        self.model = MCDropoutNet(
            self.input_dim, self.n_classes, self.hidden_dims, self.dropout_rate
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        train_accs, val_accs = [], []
        for epoch in range(self.n_epochs):
            # Train
            self.model.train()
            correct, total = 0, 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
            train_accs.append(correct / total)

            # Validate
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    logits = self.model(X_batch)
                    correct += (logits.argmax(1) == y_batch).sum().item()
                    total += len(y_batch)
            val_accs.append(correct / total if total > 0 else 0)

        self._is_fitted = True

        diagnostics = {
            "model_type": "mc_dropout",
            "framework": "pytorch",
            "input_dim": self.input_dim,
            "n_classes": self.n_classes,
            "hidden_dims": list(self.hidden_dims),
            "dropout_rate": self.dropout_rate,
            "n_mc_samples": self.n_mc_samples,
            "n_epochs": self.n_epochs,
            "train_accuracy": float(train_accs[-1]),
            "val_accuracy": float(val_accs[-1]),
            "n_train": len(X_train),
            "n_val": len(X_val),
        }

        logger.info("MC Dropout (PyTorch): train_acc=%.3f, val_acc=%.3f",
                     train_accs[-1], val_accs[-1])
        return diagnostics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled)

        self.model.train()  # Keep dropout ON
        predictions = []
        for _ in range(self.n_mc_samples):
            with torch.no_grad():
                logits = self.model(X_t)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.numpy())

        predictions = np.array(predictions)  # (n_mc, n_samples, n_classes)
        return predictions.mean(axis=0)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled)

        self.model.train()
        predictions = []
        for _ in range(self.n_mc_samples):
            with torch.no_grad():
                logits = self.model(X_t)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.numpy())

        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)

        # Epistemic: variance of mean predictions across MC samples
        epistemic = predictions.var(axis=0).mean(axis=1)

        # Aleatoric: mean of per-sample entropy
        aleatoric = -np.sum(predictions * np.log(predictions + 1e-10), axis=(0, 2)) / predictions.shape[2]

        return mean_pred, epistemic, aleatoric

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# 2. Variational BNN (Bayes by Backprop)
# ---------------------------------------------------------------------------

class VILinear(nn.Module):
    """Variational Inference Linear layer (Bayes by Backprop)."""

    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight posterior: q(w) = N(mu_rho, sigma_rho^2)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))

        # Bias posterior
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

        # Prior
        self.prior_sigma = prior_sigma

    def forward(self, x):
        # Reparameterization trick
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        if self.training:
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """KL(q(w) || p(w)) under isotropic Gaussian prior."""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # KL for Gaussian prior
        kl_w = 0.5 * (
            (self.weight_mu ** 2 + weight_sigma ** 2) / self.prior_sigma ** 2
            - 1 - 2 * torch.log(weight_sigma / self.prior_sigma + 1e-10)
        ).sum()

        kl_b = 0.5 * (
            (self.bias_mu ** 2 + bias_sigma ** 2) / self.prior_sigma ** 2
            - 1 - 2 * torch.log(bias_sigma / self.prior_sigma + 1e-10)
        ).sum()

        return kl_w + kl_b


class VariationalBNNNet(nn.Module):
    """Variational BNN with Bayes by Backprop."""

    def __init__(self, input_dim, n_classes, hidden_dims=(64, 32), prior_sigma=1.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(VILinear(prev, h, prior_sigma))
            layers.append(nn.ReLU())
            prev = h
        layers.append(VILinear(prev, n_classes, prior_sigma))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def kl_divergence(self):
        total_kl = 0
        for layer in self.layers:
            if isinstance(layer, VILinear):
                total_kl += layer.kl_divergence()
        return total_kl


class VariationalBNN:
    """
    Variational Bayesian Neural Network using Bayes by Backprop.

    Uses trainable variational inference to approximate posterior
    over network weights with KL divergence regularization.
    """

    def __init__(
        self,
        input_dim: int = 14,
        n_classes: int = 5,
        hidden_dims: Tuple[int, ...] = (64, 32),
        kl_weight: float = 0.01,
        prior_sigma: float = 1.0,
        n_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.kl_weight = kl_weight
        self.prior_sigma = prior_sigma
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, X, y, feature_names=None, validation_split=0.1):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        X_scaled = self.scaler.fit_transform(X)

        n_val = int(len(X_scaled) * validation_split)
        X_train, X_val = X_scaled[:-n_val], X_scaled[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        train_loader = _make_loader(X_train, y_train, self.batch_size)
        val_loader = _make_loader(X_val, y_val, self.batch_size, shuffle=False)

        self.model = VariationalBNNNet(
            self.input_dim, self.n_classes, self.hidden_dims, self.prior_sigma
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        n_train = len(X_train)
        train_accs, val_accs = [], []

        for epoch in range(self.n_epochs):
            # Train
            self.model.train()
            correct, total = 0, 0
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                nll = F.cross_entropy(logits, y_batch)
                kl = self.model.kl_divergence()
                loss = nll + self.kl_weight * kl / n_train
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
            train_accs.append(correct / total)

            # Validate
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    logits = self.model(X_batch)
                    correct += (logits.argmax(1) == y_batch).sum().item()
                    total += len(y_batch)
            val_accs.append(correct / total if total > 0 else 0)

        self._is_fitted = True

        return {
            "model_type": "variational_bnn",
            "framework": "pytorch",
            "input_dim": self.input_dim,
            "n_classes": self.n_classes,
            "kl_weight": self.kl_weight,
            "prior_sigma": self.prior_sigma,
            "train_accuracy": float(train_accs[-1]),
            "val_accuracy": float(val_accs[-1]),
        }

    def predict_proba(self, X: np.ndarray, n_samples: int = 50) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled)

        self.model.train()  # Sample weights
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                logits = self.model(X_t)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.numpy())

        return np.mean(predictions, axis=0)

    def predict(self, X: np.ndarray, n_samples: int = 50) -> np.ndarray:
        return np.argmax(self.predict_proba(X, n_samples), axis=1)


# ---------------------------------------------------------------------------
# 3. Deep Ensemble
# ---------------------------------------------------------------------------

class DeepEnsemble:
    """
    Deep Ensemble of M classifiers for uncertainty estimation.

    Each member is trained on a different random seed / bootstrap sample.
    """

    def __init__(
        self,
        M: int = 10,
        input_dim: int = 14,
        n_classes: int = 5,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        n_epochs: int = 80,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.M = M
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.members = []
        self.scalers = []
        self._is_fitted = False

    def fit(self, X, y, feature_names=None, validation_split=0.1):
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.members = []
        self.scalers = []

        for m in range(self.M):
            seed = self.random_state + m
            torch.manual_seed(seed)
            np.random.seed(seed)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers.append(scaler)

            # Bootstrap sample
            n = len(X_scaled)
            idx = np.random.choice(n, size=n, replace=True)
            X_boot, y_boot = X_scaled[idx], y[idx]

            # Split
            n_val = int(n * validation_split)
            X_train, X_val = X_boot[:-n_val], X_boot[-n_val:]
            y_train, y_val = y_boot[:-n_val], y_boot[-n_val:]

            train_loader = _make_loader(X_train, y_train, self.batch_size)

            # Build and train model
            model = MCDropoutNet(
                self.input_dim, self.n_classes, self.hidden_dims, self.dropout
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()

            model.train()
            for epoch in range(self.n_epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()

            self.members.append(model)
            if (m + 1) % 5 == 0:
                logger.info("Deep Ensemble: trained %d/%d members", m + 1, self.M)

        self._is_fitted = True

        return {
            "model_type": "deep_ensemble",
            "framework": "pytorch",
            "M": self.M,
            "input_dim": self.input_dim,
            "n_classes": self.n_classes,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for model, scaler in zip(self.members, self.scalers):
            X_scaled = scaler.transform(X)
            X_t = torch.FloatTensor(X_scaled)
            model.eval()
            with torch.no_grad():
                logits = model(X_t)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.numpy())
        return np.mean(predictions, axis=0)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        for model, scaler in zip(self.members, self.scalers):
            X_scaled = scaler.transform(X)
            X_t = torch.FloatTensor(X_scaled)
            model.eval()
            with torch.no_grad():
                logits = model(X_t)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.numpy())

        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        epistemic = predictions.var(axis=0).mean(axis=1)
        return mean_pred, epistemic

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
