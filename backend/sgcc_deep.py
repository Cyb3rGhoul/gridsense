from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from torch.utils.data import DataLoader, Dataset


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    padded = np.pad(arr, ((0, 0), (1, 0)), mode="constant")
    csum = np.cumsum(padded, axis=1)
    out = np.empty_like(arr)
    prefix = min(window - 1, arr.shape[1])
    if prefix > 0:
        denom = np.arange(1, prefix + 1, dtype=np.float32)
        out[:, :prefix] = csum[:, 1 : prefix + 1] / denom
    if arr.shape[1] >= window:
        out[:, window - 1 :] = (csum[:, window:] - csum[:, :-window]) / float(window)
    return out


def _robust_scale(log_values: np.ndarray) -> np.ndarray:
    med = np.median(log_values, axis=1, keepdims=True)
    q25 = np.quantile(log_values, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(log_values, 0.75, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-3)
    scaled = (log_values - med) / iqr
    return np.clip(scaled, -8.0, 8.0)


def _prepare_sequence_tensor(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    missing_mask = np.isnan(values).astype(np.float32)
    med = np.nanmedian(values, axis=1, keepdims=True)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isnan(values), med, values)
    filled = np.clip(filled, 0.0, None).astype(np.float32)
    log_values = np.log1p(filled)
    norm = _robust_scale(log_values)
    diffs = np.diff(norm, axis=1, prepend=norm[:, :1])
    seasonal = norm - _moving_average(norm, 7)
    channels = np.stack([norm, diffs, seasonal, missing_mask], axis=1).astype(np.float32)
    return channels, filled


def _autocorr_feature(arr: np.ndarray, lag: int) -> np.ndarray:
    if arr.shape[1] <= lag:
        return np.zeros(arr.shape[0], dtype=np.float32)
    left = arr[:, :-lag]
    right = arr[:, lag:]
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numer = np.sum(left_centered * right_centered, axis=1)
    denom = np.sqrt(np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1))
    return np.nan_to_num(numer / np.maximum(denom, 1e-6), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _max_zero_run(mask: np.ndarray) -> np.ndarray:
    runs = []
    for row in mask:
        if not row.any():
            runs.append(0.0)
            continue
        changes = np.diff(np.concatenate(([0], row.astype(int), [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        runs.append(float((ends - starts).max()) if len(starts) else 0.0)
    return np.asarray(runs, dtype=np.float32)


def _build_aux_features(values: np.ndarray, filled: np.ndarray, ordered_dates: pd.DatetimeIndex) -> np.ndarray:
    log_values = np.log1p(filled)
    mean = filled.mean(axis=1)
    std = filled.std(axis=1)
    median = np.median(filled, axis=1)
    q25 = np.quantile(filled, 0.25, axis=1)
    q75 = np.quantile(filled, 0.75, axis=1)
    q10 = np.quantile(filled, 0.10, axis=1)
    q90 = np.quantile(filled, 0.90, axis=1)
    zero_rate = (filled <= 0.001).mean(axis=1)
    missing_rate = np.isnan(values).mean(axis=1)
    recent_30 = filled[:, -30:].mean(axis=1)
    prev_180 = filled[:, -210:-30].mean(axis=1)
    first_half = filled[:, : filled.shape[1] // 2].mean(axis=1)
    second_half = filled[:, filled.shape[1] // 2 :].mean(axis=1)
    weekly_ma = _moving_average(filled, 7)
    recent_90 = filled[:, -90:].mean(axis=1)
    max_zero = _max_zero_run(filled <= 0.001)
    ac7 = _autocorr_feature(log_values, 7)
    ac30 = _autocorr_feature(log_values, 30)
    ac90 = _autocorr_feature(log_values, 90)
    periods = pd.Series(ordered_dates).dt.to_period("M").astype(str).to_numpy()
    monthly_means = []
    for period in sorted(set(periods)):
        mask = periods == period
        monthly_means.append(filled[:, mask].mean(axis=1))
    month_matrix = np.vstack(monthly_means).T
    week_mask = (pd.Series(ordered_dates).dt.dayofweek >= 5).to_numpy()
    weekday_mask = ~week_mask
    weekend_mean = filled[:, week_mask].mean(axis=1) if week_mask.any() else np.zeros(len(filled))
    weekday_mean = filled[:, weekday_mask].mean(axis=1) if weekday_mask.any() else np.zeros(len(filled))
    aux = np.column_stack(
        [
            mean,
            std,
            median,
            q10,
            q25,
            q75,
            q90,
            filled.max(axis=1),
            filled.min(axis=1),
            std / np.maximum(mean, 1e-3),
            zero_rate,
            missing_rate,
            recent_30,
            prev_180,
            recent_30 / np.maximum(prev_180, 1e-3),
            second_half / np.maximum(first_half, 1e-3),
            recent_90 / np.maximum(mean, 1e-3),
            np.abs(np.diff(filled, axis=1)).mean(axis=1),
            np.diff(filled, axis=1).std(axis=1),
            max_zero,
            ac7,
            ac30,
            ac90,
            (filled - weekly_ma).std(axis=1),
            month_matrix.std(axis=1),
            month_matrix.min(axis=1) / np.maximum(month_matrix.max(axis=1), 1e-3),
            weekend_mean,
            weekday_mean,
            weekend_mean / np.maximum(weekday_mean, 1e-3),
        ]
    ).astype(np.float32)
    return np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0)


class SGCCDataset(Dataset):
    def __init__(self, inputs: np.ndarray, aux: np.ndarray, labels: np.ndarray):
        self.inputs = torch.from_numpy(inputs)
        self.aux = torch.from_numpy(aux.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.aux[idx], self.labels[idx]


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.se = SqueezeExcite1D(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.se(out)
        return self.act(x + out)


class AttentionPool1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        return torch.sum(x * weights, dim=-1)


class SGCCTemporalNet(nn.Module):
    def __init__(self, in_channels: int = 4, aux_features: int = 0):
        super().__init__()
        self.aux_features = aux_features
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.stage1 = nn.Sequential(
            DilatedResidualBlock(64, 1, 0.10),
            DilatedResidualBlock(64, 2, 0.10),
            DilatedResidualBlock(64, 4, 0.10),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )
        self.stage2 = nn.Sequential(
            DilatedResidualBlock(128, 1, 0.12),
            DilatedResidualBlock(128, 2, 0.12),
            DilatedResidualBlock(128, 4, 0.12),
            DilatedResidualBlock(128, 8, 0.12),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.SiLU(),
        )
        self.stage3 = nn.Sequential(
            DilatedResidualBlock(192, 1, 0.15),
            DilatedResidualBlock(192, 2, 0.15),
            DilatedResidualBlock(192, 4, 0.15),
            DilatedResidualBlock(192, 8, 0.15),
            DilatedResidualBlock(192, 16, 0.15),
        )
        self.attn_pool = AttentionPool1D(192)
        self.raw_wide_mlp = nn.Sequential(
            nn.Linear(104, 96),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 64),
            nn.SiLU(),
        )
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_features, 256),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(192 * 3 + 64 + 128, 384),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(384, 128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        raw = x[:, :1]
        raw_wide = torch.cat(
            [
                F.adaptive_avg_pool1d(raw, 52).flatten(1),
                F.adaptive_max_pool1d(raw, 52).flatten(1),
            ],
            dim=1,
        )
        raw_wide = self.raw_wide_mlp(raw_wide)
        if aux is None:
            aux = torch.zeros((x.shape[0], self.aux_features), device=x.device, dtype=x.dtype)
        aux_wide = self.aux_mlp(aux)

        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.down1(feat)
        feat = self.stage2(feat)
        feat = self.down2(feat)
        feat = self.stage3(feat)

        pooled = torch.cat(
            [
                self.attn_pool(feat),
                F.adaptive_avg_pool1d(feat, 1).flatten(1),
                F.adaptive_max_pool1d(feat, 1).flatten(1),
                raw_wide,
                aux_wide,
            ],
            dim=1,
        )
        return self.head(pooled).squeeze(1)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        return loss.mean()


@dataclass
class TrainArtifacts:
    model: SGCCTemporalNet
    threshold: float
    val_pr_auc: float
    history: list[dict]


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1_curve = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0
    best_idx = int(np.nanargmax(f1_curve[:-1]))
    return (
        float(thresholds[best_idx]),
        float(f1_curve[best_idx]),
        float(precision[best_idx]),
        float(recall[best_idx]),
    )


def _predict_proba(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for xb, auxb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            auxb = auxb.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(xb, auxb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def _build_global_window_importance(
    model: nn.Module,
    sample_inputs: np.ndarray,
    sample_aux: np.ndarray,
    ordered_dates: pd.DatetimeIndex,
    device: torch.device,
    amp_enabled: bool,
) -> list[dict]:
    if len(sample_inputs) == 0:
        return []
    windows = []
    starts = list(range(0, sample_inputs.shape[-1], 30))
    base = torch.from_numpy(sample_inputs).to(device)
    aux = torch.from_numpy(sample_aux.astype(np.float32)).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            base_score = torch.sigmoid(model(base, aux)).cpu().numpy()
    for start in starts:
        end = min(start + 30, sample_inputs.shape[-1])
        occluded = sample_inputs.copy()
        occluded[:, 0, start:end] = 0.0
        occluded[:, 1, start:end] = 0.0
        occluded[:, 2, start:end] = 0.0
        tens = torch.from_numpy(occluded).to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                score = torch.sigmoid(model(tens, aux)).cpu().numpy()
        delta = float(np.mean(np.abs(base_score - score)))
        label = f"{ordered_dates[start].date()} to {ordered_dates[end - 1].date()}"
        windows.append({"feature": label, "importance": round(delta, 4)})
    windows.sort(key=lambda item: item["importance"], reverse=True)
    return windows[:15]


def train_sgcc_deep_model(
    values: np.ndarray,
    labels: pd.Series,
    ordered_dates: pd.DatetimeIndex,
    model_path: Path,
    device: torch.device | None = None,
    seed: int = 23,
) -> tuple[TrainArtifacts, dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    channels, filled_values = _prepare_sequence_tensor(values)
    aux_features = _build_aux_features(values, filled_values, ordered_dates)
    y = labels.to_numpy(dtype=np.int64)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, stratify=y[train_idx], random_state=seed + 1)

    from backend.pipeline import _extract_sgcc_features

    rich_feature_frame = _extract_sgcc_features(values, ordered_dates)
    rich_features = rich_feature_frame.to_numpy(dtype=np.float32)

    train_inputs = channels[train_idx]
    val_inputs = channels[val_idx]
    test_inputs = channels[test_idx]
    train_aux = aux_features[train_idx]
    val_aux = aux_features[val_idx]
    test_aux = aux_features[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    variance_selector = VarianceThreshold(threshold=1e-5)
    train_rich = variance_selector.fit_transform(rich_features[train_idx])
    val_rich = variance_selector.transform(rich_features[val_idx])
    test_rich = variance_selector.transform(rich_features[test_idx])
    kbest = SelectKBest(f_classif, k=min(96, train_rich.shape[1]))
    train_rich = kbest.fit_transform(train_rich, y_train).astype(np.float32)
    val_rich = kbest.transform(val_rich).astype(np.float32)
    test_rich = kbest.transform(test_rich).astype(np.float32)
    train_aux = np.concatenate([train_aux, train_rich], axis=1)
    val_aux = np.concatenate([val_aux, val_rich], axis=1)
    test_aux = np.concatenate([test_aux, test_rich], axis=1)

    aux_mean = train_aux.mean(axis=0, keepdims=True)
    aux_std = np.maximum(train_aux.std(axis=0, keepdims=True), 1e-3)
    train_aux = (train_aux - aux_mean) / aux_std
    val_aux = (val_aux - aux_mean) / aux_std
    test_aux = (test_aux - aux_mean) / aux_std

    train_ds = SGCCDataset(train_inputs, train_aux, y_train)
    val_ds = SGCCDataset(val_inputs, val_aux, y_val)
    test_ds = SGCCDataset(test_inputs, test_aux, y_test)

    class_counts = np.bincount(y_train)
    sample_weights = np.where(y_train == 1, class_counts[0] / max(class_counts[1], 1), 1.0).astype(np.float64)
    batch_size = 256 if device.type == "cuda" else 96
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=amp_enabled)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=amp_enabled)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=amp_enabled)

    model = SGCCTemporalNet(in_channels=train_inputs.shape[1], aux_features=train_aux.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    pos_weight = torch.tensor([class_counts[0] / max(class_counts[1], 1)], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_state = None
    best_val_ap = -math.inf
    best_threshold = 0.5
    history: list[dict] = []
    patience = 5
    stagnant = 0
    max_epochs = 18

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, auxb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            auxb = auxb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(xb, auxb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            batch = xb.shape[0]
            running_loss += float(loss.detach().cpu()) * batch
            seen += batch

        val_proba, val_true = _predict_proba(model, val_loader, device, amp_enabled)
        val_ap = float(average_precision_score(val_true, val_proba))
        val_threshold, val_f1, val_precision, val_recall = _best_threshold(val_true, val_proba)
        scheduler.step(val_ap)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(seen, 1),
                "val_pr_auc": val_ap,
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall,
            }
        )
        print(
            f"epoch {epoch:02d} | loss {running_loss / max(seen, 1):.4f} | "
            f"val_pr_auc {val_ap:.4f} | val_f1 {val_f1:.4f}"
        )

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_threshold = val_threshold
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "threshold": best_threshold,
            "val_pr_auc": best_val_ap,
            "history": history,
            "aux_mean": aux_mean,
            "aux_std": aux_std,
            "ordered_dates": [str(x.date()) for x in ordered_dates],
        },
        model_path,
    )

    test_proba, test_true = _predict_proba(model, test_loader, device, amp_enabled)
    pred = (test_proba >= best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(test_true, pred, average="binary", zero_division=0)

    top_indices = np.argsort(-test_proba)[:12]
    cases = []
    for pos in top_indices:
        raw_seq = filled_values[test_idx[pos]]
        recent_30 = float(np.mean(raw_seq[-30:]))
        prev_180 = float(np.mean(raw_seq[-210:-30])) if raw_seq.shape[0] >= 210 else float(np.mean(raw_seq[:-30]))
        drop_pct = max(0.0, (1.0 - recent_30 / max(prev_180, 1e-3)) * 100.0)
        cases.append(
            {
                "row_index": int(test_idx[pos]),
                "label": int(test_true[pos]),
                "theft_probability": round(float(test_proba[pos]), 4),
                "recent_drop_pct": round(drop_pct, 1),
                "missing_rate": round(float(np.isnan(values[test_idx[pos]]).mean()), 3),
                "zero_rate": round(float((np.clip(values[test_idx[pos]], 0, None) <= 0.001).mean()), 3),
                "explanation": (
                    f"Sequence model flags a sustained shape change with probability {test_proba[pos]:.2f}; "
                    f"recent 30-day usage is down {drop_pct:.0f}% versus the prior baseline."
                ),
            }
        )

    importance_source = test_inputs[: min(512, len(test_inputs))]
    importance_aux = test_aux[: min(512, len(test_aux))]
    feature_importance = _build_global_window_importance(model, importance_source, importance_aux, ordered_dates, device, amp_enabled)

    metrics = {
        "threshold": round(float(best_threshold), 4),
        "roc_auc": round(float(roc_auc_score(test_true, test_proba)), 4),
        "pr_auc": round(float(average_precision_score(test_true, test_proba)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(test_true, pred).tolist(),
        "roc_curve": [
            {"fpr": round(float(x), 4), "tpr": round(float(yv), 4)}
            for x, yv in zip(*[arr[:: max(1, len(arr) // 80)] for arr in roc_curve(test_true, test_proba)[:2]])
        ],
        "pr_curve": [
            {"recall": round(float(x), 4), "precision": round(float(yv), 4)}
            for x, yv in zip(*[arr[:: max(1, len(arr) // 80)] for arr in precision_recall_curve(test_true, test_proba)[:2][::-1]])
        ],
        "feature_importance": feature_importance,
        "top_cases": cases,
        "component_metrics": {
            "deep_temporal_net": {
                "val_pr_auc": round(float(best_val_ap), 4),
                "test_pr_auc": round(float(average_precision_score(test_true, test_proba)), 4),
                "test_roc_auc": round(float(roc_auc_score(test_true, test_proba)), 4),
            }
        },
        "training_history": history,
        "device": str(device),
    }
    return TrainArtifacts(model=model, threshold=best_threshold, val_pr_auc=best_val_ap, history=history), metrics


def build_sgcc_theft_validation_deep(path: Path, model_path: Path) -> dict:
    if not path.exists():
        return {"available": False, "reason": "SGCC split archive has not been extracted yet."}

    raw = pd.read_csv(path)
    labels = raw["FLAG"].astype(int)
    date_cols = [col for col in raw.columns if col not in {"CONS_NO", "FLAG"}]
    ordered_cols = sorted(date_cols, key=lambda c: pd.to_datetime(c, format="%Y/%m/%d"))
    ordered_dates = pd.to_datetime(ordered_cols, format="%Y/%m/%d")
    values = raw[ordered_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid_rows = ~np.isnan(values).all(axis=1)
    raw = raw.loc[valid_rows].reset_index(drop=True)
    labels = labels.loc[valid_rows].reset_index(drop=True)
    values = values[valid_rows]

    _, metrics = train_sgcc_deep_model(values, labels, ordered_dates, model_path=model_path)
    metrics.update(
        {
            "available": True,
            "dataset": "SGCC Electricity Theft Detection",
            "customers": int(len(raw)),
            "days": int(len(ordered_cols)),
            "positive_rate": round(float(labels.mean()), 4),
            "model": "GPU temporal deep net with dilated residual CNN, attention pooling, and focal loss",
        }
    )

    for case in metrics.get("top_cases", []):
        idx = case.pop("row_index")
        case["consumer_id"] = raw.iloc[idx]["CONS_NO"]
    return metrics
