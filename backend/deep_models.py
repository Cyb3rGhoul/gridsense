"""CPU-friendly deep temporal models for SGCC theft detection.

Two architectures, both used as additional base learners in the existing
tabular stacking ensemble:

- SGCCConvNet : 1D-CNN over the raw consumption sequence (3 channels:
  log-normalized consumption, first differences, missing-mask).
- SGCCGRUNet  : bidirectional GRU over weekly aggregates (5 stats per week).

Both can take the engineered tabular features as auxiliary input fused
into the head. Trained with focal loss + class-balanced sampling.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def _set_threads() -> None:
    threads = int(os.environ.get("SGCC_TORCH_THREADS", os.cpu_count() or 4))
    torch.set_num_threads(threads)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def prepare_cnn_inputs(values: np.ndarray) -> np.ndarray:
    """(N, 3, T) tensor: log-scaled consumption, first differences, missing mask."""
    missing_mask = np.isnan(values).astype(np.float32)
    med = np.nanmedian(values, axis=1, keepdims=True)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isnan(values), med, values)
    filled = np.clip(filled, 0.0, None).astype(np.float32)
    log_vals = np.log1p(filled)

    med_log = np.median(log_vals, axis=1, keepdims=True)
    q25 = np.quantile(log_vals, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(log_vals, 0.75, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-3)
    scaled = np.clip((log_vals - med_log) / iqr, -8.0, 8.0)
    diffs = np.diff(scaled, axis=1, prepend=scaled[:, :1])
    return np.stack([scaled, diffs, missing_mask], axis=1).astype(np.float32)


def prepare_gru_inputs(values: np.ndarray) -> np.ndarray:
    """(N, weeks, 5) tensor of weekly aggregates."""
    filled = np.nan_to_num(np.clip(values, 0, None), nan=0.0).astype(np.float32)
    n_full_weeks = filled.shape[1] // 7
    weekly = filled[:, : n_full_weeks * 7].reshape(filled.shape[0], n_full_weeks, 7)
    feat = np.stack(
        [
            weekly.mean(axis=2),
            weekly.std(axis=2),
            weekly.max(axis=2),
            weekly.min(axis=2),
            (weekly <= 0.001).mean(axis=2),
        ],
        axis=2,
    ).astype(np.float32)
    fmean = feat.mean(axis=1, keepdims=True)
    fstd = np.maximum(feat.std(axis=1, keepdims=True), 1e-3)
    return np.clip((feat - fmean) / fstd, -8.0, 8.0)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


class SGCCConvNet(nn.Module):
    """CPU-tuned 1D-CNN. Aggressive downsampling so the inner convs see a
    short sequence: 1034 -> 259 (stride-4 stem) -> 130 (pool) -> 65 (pool).
    """

    def __init__(self, in_channels: int = 3, n_aux: int = 0):
        super().__init__()
        self.n_aux = n_aux
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        if n_aux > 0:
            self.aux_mlp = nn.Sequential(
                nn.Linear(n_aux, 96),
                nn.ReLU(inplace=True),
                nn.Dropout(0.30),
                nn.Linear(96, 48),
                nn.ReLU(inplace=True),
            )
            head_in = 64 + 48
        else:
            self.aux_mlp = None
            head_in = 64
        self.head = nn.Sequential(
            nn.Linear(head_in, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.pool(self.conv(x)).flatten(1)
        if self.aux_mlp is not None and aux is not None:
            feat = torch.cat([feat, self.aux_mlp(aux)], dim=1)
        return self.head(feat).squeeze(-1)


class SGCCGRUNet(nn.Module):
    def __init__(self, n_features: int = 5, n_aux: int = 0):
        super().__init__()
        self.n_aux = n_aux
        self.gru = nn.GRU(
            n_features, 64, num_layers=2,
            batch_first=True, dropout=0.20, bidirectional=True,
        )
        if n_aux > 0:
            self.aux_mlp = nn.Sequential(
                nn.Linear(n_aux, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.30),
            )
            head_in = 128 + 64
        else:
            self.aux_mlp = None
            head_in = 128
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.gru(x)
        feat = out[:, -1, :]
        if self.aux_mlp is not None and aux is not None:
            feat = torch.cat([feat, self.aux_mlp(aux)], dim=1)
        return self.head(feat).squeeze(-1)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


class _SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, aux: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.aux = torch.from_numpy(aux.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.aux[idx], self.y[idx]


def _train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    best_ap = -1.0
    best_state = None
    stagnant = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, auxb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, auxb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()) * xb.shape[0]
            n_seen += xb.shape[0]

        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for xb, auxb, yb in val_loader:
                probs.append(torch.sigmoid(model(xb, auxb)).numpy())
                labels.append(yb.numpy())
        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
        val_ap = float(average_precision_score(labels, probs))
        scheduler.step(val_ap)
        print(
            f"      epoch {epoch:02d} train_loss={running / max(n_seen, 1):.4f} val_AP={val_ap:.4f}"
        )
        if val_ap > best_ap:
            best_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ap


def train_oof_deep(
    model_factory: Callable[[], nn.Module],
    X_seq: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cv: StratifiedKFold,
    name: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """K-fold OOF training for a deep base model.

    Returns (oof_train, test_pred) where oof_train.shape == (len(train_idx),)
    and test_pred is the average of fold-models' predictions on the test set.
    """
    _set_threads()
    n_train = len(train_idx)
    n_test = len(test_idx)
    oof = np.zeros(n_train, dtype=np.float32)
    test_pred = np.zeros(n_test, dtype=np.float32)
    n_folds = cv.get_n_splits()
    print(f"  {name}: {n_folds}-fold OOF, max_epochs={max_epochs}, batch_size={batch_size}")

    y_train_local = y[train_idx]

    for fold_idx, (fit_local, val_local) in enumerate(
        cv.split(np.arange(n_train), y_train_local), start=1
    ):
        print(f"    fold {fold_idx}/{n_folds}")
        X_fit = X_seq[train_idx][fit_local]
        aux_fit = aux[train_idx][fit_local]
        y_fit = y_train_local[fit_local]
        X_val = X_seq[train_idx][val_local]
        aux_val = aux[train_idx][val_local]
        y_val = y_train_local[val_local]
        X_te = X_seq[test_idx]
        aux_te = aux[test_idx]
        y_te = y[test_idx]

        cnt = np.bincount(y_fit.astype(int))
        weights = np.where(y_fit == 1, cnt[0] / max(cnt[1], 1), 1.0).astype(np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_ds = _SeqDataset(X_fit, aux_fit, y_fit)
        val_ds = _SeqDataset(X_val, aux_val, y_val)
        test_ds = _SeqDataset(X_te, aux_te, y_te)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        torch.manual_seed(fold_idx * 7 + 23)
        model = model_factory()
        model, best_ap = _train_one_fold(model, train_loader, val_loader, max_epochs, patience)
        print(f"    fold {fold_idx} best_val_AP={best_ap:.4f}")

        model.eval()
        with torch.no_grad():
            val_probs = []
            for xb, auxb, _ in val_loader:
                val_probs.append(torch.sigmoid(model(xb, auxb)).numpy())
            oof[val_local] = np.concatenate(val_probs)

            test_probs = []
            for xb, auxb, _ in test_loader:
                test_probs.append(torch.sigmoid(model(xb, auxb)).numpy())
            test_pred += np.concatenate(test_probs) / n_folds

    print(
        f"  {name} OOF AP={average_precision_score(y_train_local, oof):.4f} "
        f"OOF AUC={roc_auc_score(y_train_local, oof):.4f}"
    )
    return oof, test_pred
