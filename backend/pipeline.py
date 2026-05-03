from __future__ import annotations

import json
import math
import os
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgbm
from scipy.optimize import differential_evolution

try:
    from imblearn.over_sampling import BorderlineSMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    import torch  # noqa: F401
    from backend.deep_models import (
        SGCCConvNet,
        prepare_cnn_inputs,
        train_oof_deep,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
LEAD_ZIP = ROOT / "data" / "raw" / "lead" / "data" / "lead1.0-small.zip"
INDIA_DIR = ROOT / "data" / "raw" / "kaggle" / "india"
INDIA_PRIMARY = INDIA_DIR / "SM Cleaned Data BR2019.csv"
LONDON_HH_BLOCK = ROOT / "data" / "raw" / "kaggle" / "london" / "hhblock_dataset" / "hhblock_dataset" / "block_0.csv"
LONDON_WEATHER = ROOT / "data" / "raw" / "kaggle" / "london" / "weather_hourly_darksky.csv"
SGCC_CSV = ROOT / "data" / "raw" / "sgcc_extracted" / "data.csv"


@dataclass(frozen=True)
class GridConfig:
    feeders: int = 10
    meters_per_feeder: int = 20
    days: int = 60
    interval_minutes: int = 15
    seed: int = 42


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _daily_profile(hour: np.ndarray, segment: str) -> np.ndarray:
    evening = np.exp(-((hour - 20) / 3.0) ** 2)
    morning = np.exp(-((hour - 8) / 2.4) ** 2)
    office = np.exp(-((hour - 14) / 4.5) ** 2)
    if segment == "residential":
        return 0.30 + 0.45 * morning + 1.10 * evening
    if segment == "commercial":
        return 0.25 + 1.35 * office + 0.25 * evening
    return 0.35 + 0.55 * office + 0.75 * evening


def generate_synthetic_meter_data(config: GridConfig = GridConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    periods = config.days * 24 * (60 // config.interval_minutes)
    ts = pd.date_range("2026-01-01", periods=periods, freq=f"{config.interval_minutes}min")
    rows: list[pd.DataFrame] = []
    localities = [
        "Koramangala",
        "Yelahanka",
        "Whitefield",
        "Indiranagar",
        "Electronic City",
        "Rajajinagar",
        "Jayanagar",
        "Hebbal",
        "Peenya",
        "BTM Layout",
    ]
    feeder_capacity = np.linspace(950, 1550, config.feeders)

    for feeder_id in range(config.feeders):
        segment_mix = rng.choice(
            ["residential", "commercial", "mixed"],
            size=config.meters_per_feeder,
            p=[0.52, 0.28, 0.20],
        )
        for meter_idx, segment in enumerate(segment_mix):
            meter_id = f"F{feeder_id + 1:02d}-M{meter_idx + 1:03d}"
            base = rng.uniform(0.18, 1.25) * (1.0 if segment != "commercial" else 1.7)
            hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
            weekday = ts.dayofweek.to_numpy()
            weekend = (weekday >= 5).astype(float)
            temp = 25 + 5 * np.sin(2 * np.pi * (ts.dayofyear.to_numpy() - 35) / 365) + 3 * np.sin(
                2 * np.pi * (hour - 13) / 24
            )
            weather_lift = np.maximum(temp - 29, 0) * 0.045
            profile = _daily_profile(hour, segment)
            if segment == "commercial":
                profile *= 1 - 0.35 * weekend
            else:
                profile *= 1 + 0.12 * weekend
            trend = 1 + 0.0012 * np.arange(periods) / 96
            noise = rng.normal(0, 0.08, periods)
            kwh = np.clip(base * profile * trend * (1 + weather_lift + noise), 0.02, None)

            theft = 0
            event_type = "normal"
            if rng.random() < 0.11:
                theft = 1
                start = rng.integers(periods // 3, periods - 96 * 7)
                attack = rng.choice(["sudden_drop", "night_dips", "repeated_readings"])
                event_type = attack
                if attack == "sudden_drop":
                    kwh[start:] *= rng.uniform(0.28, 0.55)
                elif attack == "night_dips":
                    mask = (np.arange(periods) >= start) & ((hour < 5) | (hour > 22))
                    kwh[mask] *= rng.uniform(0.05, 0.25)
                else:
                    for block in range(start, periods, 96 * 3):
                        kwh[block : block + 12] = np.round(kwh[block], 3)

            missing = rng.random(periods) < 0.006
            kwh[missing] = np.nan
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": ts,
                        "meter_id": meter_id,
                        "feeder_id": f"F{feeder_id + 1:02d}",
                        "locality": localities[feeder_id],
                        "segment": segment,
                        "capacity_kw": feeder_capacity[feeder_id],
                        "temperature_c": np.round(temp, 2),
                        "humidity": np.round(58 + 18 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 4, periods), 2),
                        "kwh": np.round(kwh, 4),
                        "label_theft": theft,
                        "event_type": event_type,
                    }
                )
            )

    df = pd.concat(rows, ignore_index=True)
    df["kwh"] = df.groupby("meter_id")["kwh"].transform(lambda s: s.interpolate().bfill().ffill())
    return df


def load_lead_real_data(path: Path = LEAD_ZIP) -> pd.DataFrame:
    """Load the real LEAD 1.0 small dataset and map it to utility-style fields."""
    with zipfile.ZipFile(path) as archive:
        raw = pd.read_csv(archive.open("lead1.0-small.csv"))
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.sort_values(["building_id", "timestamp"])
    raw["meter_reading"] = raw.groupby("building_id")["meter_reading"].transform(lambda s: s.interpolate().bfill().ffill())
    raw["meter_reading"] = raw["meter_reading"].clip(lower=0)

    buildings = sorted(raw["building_id"].unique())
    feeder_map = {building_id: f"F{idx % 10 + 1:02d}" for idx, building_id in enumerate(buildings)}
    localities = [
        "Koramangala",
        "Yelahanka",
        "Whitefield",
        "Indiranagar",
        "Electronic City",
        "Rajajinagar",
        "Jayanagar",
        "Hebbal",
        "Peenya",
        "BTM Layout",
    ]
    segment_names = ["commercial", "mixed", "residential"]
    raw["meter_id"] = raw["building_id"].map(lambda x: f"LEAD-{int(x):04d}")
    raw["feeder_id"] = raw["building_id"].map(feeder_map)
    raw["locality"] = raw["feeder_id"].map(lambda x: localities[int(x[1:]) - 1])
    raw["segment"] = raw["building_id"].map(lambda x: segment_names[int(x) % len(segment_names)])
    raw["kwh"] = raw["meter_reading"]
    raw["label_theft"] = raw["anomaly"].astype(int)
    raw["event_type"] = np.where(raw["label_theft"] == 1, "real_LEAD_anomaly", "normal")

    hour = raw["timestamp"].dt.hour + raw["timestamp"].dt.minute / 60
    day = raw["timestamp"].dt.dayofyear
    raw["temperature_c"] = np.round(25 + 5 * np.sin(2 * np.pi * (day - 35) / 365) + 3 * np.sin(2 * np.pi * (hour - 13) / 24), 2)
    raw["humidity"] = np.round(58 + 18 * np.sin(2 * np.pi * hour / 24), 2)
    feeder_peak = raw.groupby("feeder_id")["kwh"].quantile(0.98).mul(1.25).clip(lower=1)
    raw["capacity_kw"] = raw["feeder_id"].map(feeder_peak)
    return raw[
        [
            "timestamp",
            "meter_id",
            "feeder_id",
            "locality",
            "segment",
            "capacity_kw",
            "temperature_c",
            "humidity",
            "kwh",
            "label_theft",
            "event_type",
        ]
    ]


def load_india_real_data(path: Path = INDIA_PRIMARY, max_days: int = 75) -> pd.DataFrame:
    """Load real CEEW/Kaggle Indian smart-meter data and map it to GridSense fields."""
    usecols = ["x_Timestamp", "t_kWh", "z_Avg Voltage (Volt)", "z_Avg Current (Amp)", "y_Freq (Hz)", "meter"]
    chunks: list[pd.DataFrame] = []
    min_ts: pd.Timestamp | None = None
    cutoff: pd.Timestamp | None = None
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=600_000):
        chunk["x_Timestamp"] = pd.to_datetime(chunk["x_Timestamp"], errors="coerce")
        chunk = chunk.dropna(subset=["x_Timestamp", "meter", "t_kWh"])
        if min_ts is None:
            min_ts = chunk["x_Timestamp"].min()
            cutoff = min_ts + pd.Timedelta(days=max_days)
        chunk = chunk[chunk["x_Timestamp"] <= cutoff]
        if not chunk.empty:
            chunks.append(chunk)
        if cutoff is not None and chunk["x_Timestamp"].max() >= cutoff:
            break
    raw = pd.concat(chunks, ignore_index=True)
    raw = raw.sort_values(["meter", "x_Timestamp"])
    raw["t_kWh"] = pd.to_numeric(raw["t_kWh"], errors="coerce").fillna(0).clip(lower=0)
    meter_counts = raw.groupby("meter").size()
    valid_meters = meter_counts[meter_counts >= 7 * 24 * 20].index
    raw = raw[raw["meter"].isin(valid_meters)].copy()

    meters = sorted(raw["meter"].unique())
    feeder_map = {meter: f"F{idx % 10 + 1:02d}" for idx, meter in enumerate(meters)}
    localities = [
        "Bareilly North",
        "Bareilly South",
        "Bareilly East",
        "Bareilly West",
        "Bareilly Central",
        "Mathura North",
        "Mathura South",
        "Mathura East",
        "Mathura West",
        "Mathura Central",
    ]
    raw["timestamp"] = raw["x_Timestamp"]
    raw["meter_id"] = raw["meter"].astype(str)
    raw["feeder_id"] = raw["meter"].map(feeder_map)
    raw["locality"] = raw["feeder_id"].map(lambda x: localities[int(x[1:]) - 1])
    raw["segment"] = "residential"
    raw["kwh"] = raw["t_kWh"]
    raw["label_theft"] = 0
    raw["event_type"] = "unlabelled_real_india"
    raw["temperature_c"] = 28 + 4 * np.sin(2 * np.pi * (raw["timestamp"].dt.dayofyear - 80) / 365)
    raw["humidity"] = 62 + 14 * np.sin(2 * np.pi * raw["timestamp"].dt.hour / 24)
    feeder_peak = raw.groupby("feeder_id")["kwh"].quantile(0.99).mul(20).clip(lower=1)
    raw["capacity_kw"] = raw["feeder_id"].map(feeder_peak)
    return raw[
        [
            "timestamp",
            "meter_id",
            "feeder_id",
            "locality",
            "segment",
            "capacity_kw",
            "temperature_c",
            "humidity",
            "kwh",
            "label_theft",
            "event_type",
        ]
    ]


def load_london_real_data(path: Path = LONDON_HH_BLOCK, max_days: int = 120, max_meters: int = 50) -> pd.DataFrame:
    """Load real London smart-meter half-hourly data with real hourly weather."""
    raw = pd.read_csv(path)
    meter_ids = raw["LCLid"].drop_duplicates().head(max_meters).tolist()
    raw = raw[raw["LCLid"].isin(meter_ids)].copy()
    raw["day"] = pd.to_datetime(raw["day"], errors="coerce")
    latest_day = raw["day"].max()
    raw = raw[raw["day"] >= latest_day - pd.Timedelta(days=max_days)].copy()

    hh_cols = [f"hh_{idx}" for idx in range(48)]
    long = raw.melt(id_vars=["LCLid", "day"], value_vars=hh_cols, var_name="slot", value_name="kwh")
    long["slot_idx"] = long["slot"].str.replace("hh_", "", regex=False).astype(int)
    long["timestamp"] = long["day"] + pd.to_timedelta(long["slot_idx"] * 30, unit="m")
    long["kwh"] = pd.to_numeric(long["kwh"], errors="coerce").fillna(0).clip(lower=0)

    weather = pd.read_csv(LONDON_WEATHER, usecols=["time", "temperature", "humidity"])
    weather["timestamp_hour"] = pd.to_datetime(weather["time"], errors="coerce")
    long["timestamp_hour"] = long["timestamp"].dt.floor("h")
    long = long.merge(weather[["timestamp_hour", "temperature", "humidity"]], on="timestamp_hour", how="left")
    long["temperature"] = long["temperature"].interpolate().bfill().ffill()
    long["humidity"] = long["humidity"].interpolate().bfill().ffill().mul(100)

    localities = [
        "Koramangala",
        "Yelahanka",
        "Whitefield",
        "Indiranagar",
        "Electronic City",
        "Rajajinagar",
        "Jayanagar",
        "Hebbal",
        "Peenya",
        "BTM Layout",
    ]
    meter_order = sorted(long["LCLid"].unique())
    feeder_map = {meter: f"F{idx % 10 + 1:02d}" for idx, meter in enumerate(meter_order)}
    segment_map = {meter: ["residential", "commercial", "mixed"][idx % 3] for idx, meter in enumerate(meter_order)}
    long["meter_id"] = long["LCLid"]
    long["feeder_id"] = long["LCLid"].map(feeder_map)
    long["locality"] = long["feeder_id"].map(lambda x: localities[int(x[1:]) - 1])
    long["segment"] = long["LCLid"].map(segment_map)
    long["label_theft"] = 0
    long["event_type"] = "unlabelled_real_london"
    feeder_peak = long.groupby("feeder_id")["kwh"].quantile(0.995).mul(2).mul(1.35).clip(lower=1)
    long["capacity_kw"] = long["feeder_id"].map(feeder_peak)
    long["temperature_c"] = long["temperature"]
    return long[
        [
            "timestamp",
            "meter_id",
            "feeder_id",
            "locality",
            "segment",
            "capacity_kw",
            "temperature_c",
            "humidity",
            "kwh",
            "label_theft",
            "event_type",
        ]
    ].sort_values(["meter_id", "timestamp"])


def _time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour + ts.dt.minute / 60
    out["dow"] = ts.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


def build_forecasts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    interval_minutes = int(pd.to_datetime(df["timestamp"]).sort_values().drop_duplicates().diff().dt.total_seconds().dropna().mode().iloc[0] / 60)
    intervals_per_hour = max(1, 60 // interval_minutes)
    intervals_per_day = max(1, 24 * intervals_per_hour)
    horizon = intervals_per_day
    feeder = (
        df.groupby(["timestamp", "feeder_id", "locality", "capacity_kw"], as_index=False)
        .agg(load_kw=("kwh", lambda x: x.sum() * intervals_per_hour), temperature_c=("temperature_c", "mean"), humidity=("humidity", "mean"))
        .sort_values(["feeder_id", "timestamp"])
    )
    feeder = _time_features(feeder)
    feeder["lag_1h"] = feeder.groupby("feeder_id")["load_kw"].shift(intervals_per_hour)
    feeder["lag_24h"] = feeder.groupby("feeder_id")["load_kw"].shift(intervals_per_day)
    feeder["roll_24h"] = feeder.groupby("feeder_id")["load_kw"].transform(lambda s: s.shift(1).rolling(intervals_per_day, min_periods=8).mean())
    model_frame = feeder.dropna().copy()
    features = ["hour_sin", "hour_cos", "dow", "is_weekend", "temperature_c", "humidity", "lag_1h", "lag_24h", "roll_24h"]
    cutoff = model_frame["timestamp"].quantile(0.82)
    train = model_frame[model_frame["timestamp"] <= cutoff]
    test = model_frame[model_frame["timestamp"] > cutoff]
    model = LGBMRegressor(
        n_estimators=420,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=7,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(train[features], train["load_kw"])
    joblib.dump(model, MODELS_DIR / "forecast_lgbm.joblib")
    preds = model.predict(test[features])
    baseline_preds = test["lag_24h"].to_numpy()
    abs_err = np.abs(test["load_kw"].to_numpy() - preds)
    baseline_abs_err = np.abs(test["load_kw"].to_numpy() - baseline_preds)
    mae = float(abs_err.mean())
    baseline_mae = float(baseline_abs_err.mean())
    nontrivial_floor = max(0.25, float(test["load_kw"].quantile(0.35)))
    mask = test["load_kw"].to_numpy() >= nontrivial_floor
    mape = float(np.mean(abs_err[mask] / np.maximum(test["load_kw"].to_numpy()[mask], 0.01)))
    smape = float(np.mean(2 * abs_err / np.maximum(np.abs(test["load_kw"].to_numpy()) + np.abs(preds), 0.01)))
    baseline_smape = float(np.mean(2 * baseline_abs_err / np.maximum(np.abs(test["load_kw"].to_numpy()) + np.abs(baseline_preds), 0.01)))

    latest_rows = []
    for feeder_id, hist in feeder.groupby("feeder_id"):
        hist = hist.sort_values("timestamp")
        if len(hist) < intervals_per_day + intervals_per_hour:
            continue
        feeder_rows = []
        last_ts = hist["timestamp"].max()
        last = hist.iloc[-1]
        for step in range(1, horizon + 1):
            ts = last_ts + pd.Timedelta(minutes=interval_minutes * step)
            lag_1h = hist.iloc[-intervals_per_hour]["load_kw"] if step <= intervals_per_hour else feeder_rows[-intervals_per_hour]["forecast_kw"]
            lag_24h = hist.iloc[-intervals_per_day + step - 1]["load_kw"]
            roll_24h = hist.tail(intervals_per_day)["load_kw"].mean()
            row = {
                "timestamp": ts,
                "feeder_id": feeder_id,
                "locality": last["locality"],
                "capacity_kw": float(last["capacity_kw"]),
                "temperature_c": float(last["temperature_c"] + 1.5 * math.sin(2 * math.pi * ts.hour / 24)),
                "humidity": float(last["humidity"]),
                "hour": ts.hour + ts.minute / 60,
                "dow": ts.dayofweek,
                "is_weekend": int(ts.dayofweek >= 5),
                "hour_sin": math.sin(2 * math.pi * (ts.hour + ts.minute / 60) / 24),
                "hour_cos": math.cos(2 * math.pi * (ts.hour + ts.minute / 60) / 24),
                "lag_1h": lag_1h,
                "lag_24h": lag_24h,
                "roll_24h": roll_24h,
            }
            pred = float(model.predict(pd.DataFrame([row])[features])[0])
            row["forecast_kw"] = round(pred, 2)
            row["lower_kw"] = round(pred * (1 - max(0.06, mape)), 2)
            row["upper_kw"] = round(pred * (1 + max(0.08, mape * 1.3)), 2)
            row["risk_score"] = round(min(100, 100 * row["upper_kw"] / row["capacity_kw"]), 1)
            row["risk_level"] = "critical" if row["risk_score"] >= 92 else "high" if row["risk_score"] >= 78 else "normal"
            feeder_rows.append(row)
        latest_rows.extend(feeder_rows)

    forecasts = pd.DataFrame(latest_rows)
    metrics = {
        "forecast_mape": round(mape, 4),
        "forecast_mae_kw": round(mae, 4),
        "forecast_smape": round(smape, 4),
        "forecast_baseline_smape": round(baseline_smape, 4),
        "forecast_baseline_mae_kw": round(baseline_mae, 4),
        "forecast_model": "LightGBM regressor",
        "forecast_baseline": "24-hour lag persistence baseline.",
        "test_rows": int(len(test)),
    }
    return forecasts, metrics


def build_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    daily = (
        df.assign(date=pd.to_datetime(df["timestamp"]).dt.date)
        .groupby(["meter_id", "feeder_id", "locality", "segment", "date"], as_index=False)
        .agg(daily_kwh=("kwh", "sum"), label_theft=("label_theft", "max"), event_type=("event_type", "last"))
        .sort_values(["meter_id", "date"])
    )
    daily["prev_14"] = daily.groupby("meter_id")["daily_kwh"].transform(lambda s: s.shift(1).rolling(14, min_periods=5).mean())
    daily["prev_45"] = daily.groupby("meter_id")["daily_kwh"].transform(lambda s: s.shift(1).rolling(45, min_periods=10).mean())
    peer_group_size = daily.groupby(["date", "feeder_id", "segment"])["daily_kwh"].transform("count")
    peer_segment = daily.groupby(["date", "feeder_id", "segment"])["daily_kwh"].transform("median")
    peer_feeder = daily.groupby(["date", "feeder_id"])["daily_kwh"].transform("median")
    peer_feeder_size = daily.groupby(["date", "feeder_id"])["daily_kwh"].transform("count")
    peer_global_segment = daily.groupby(["date", "segment"])["daily_kwh"].transform("median")
    peer = np.where(
        peer_group_size >= 3,
        peer_segment,
        np.where(peer_feeder_size >= 2, peer_feeder, peer_global_segment),
    )
    daily["peer_ratio"] = daily["daily_kwh"] / pd.Series(peer, index=daily.index).replace(0, np.nan)
    daily["drop_ratio"] = daily["daily_kwh"] / daily["prev_45"].replace(0, np.nan)
    daily["volatility"] = daily.groupby("meter_id")["daily_kwh"].transform(lambda s: s.rolling(7, min_periods=3).std())
    frame = daily.dropna().copy()
    features = ["daily_kwh", "prev_14", "prev_45", "peer_ratio", "drop_ratio", "volatility"]
    iso = IsolationForest(contamination=0.10, random_state=10)
    frame["iso_score"] = -iso.fit_predict(frame[features])
    has_labels = frame["label_theft"].nunique() > 1
    if has_labels:
        X_train, X_test, y_train, y_test = train_test_split(
            frame[features], frame["label_theft"], test_size=0.28, stratify=frame["label_theft"], random_state=9
        )
        clf = RandomForestClassifier(n_estimators=160, class_weight="balanced", random_state=9, n_jobs=1)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.52).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        frame["theft_probability"] = clf.predict_proba(frame[features])[:, 1]
        metrics = {
            "theft_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "theft_average_precision": round(float(average_precision_score(y_test, proba)), 4),
            "theft_precision": round(float(precision), 4),
            "theft_recall": round(float(recall), 4),
            "theft_f1": round(float(f1_score(y_test, pred)), 4),
            "anomaly_mode": "supervised labelled anomaly evaluation",
            "visible_false_positive_proxy": "Use precision plus post-inspection outcomes; LEAD anomaly labels are used when real data is active.",
        }
    else:
        frame["theft_probability"] = np.clip(
            (frame["drop_ratio"].lt(0.62).astype(float) * 0.40)
            + (frame["peer_ratio"].lt(0.55).astype(float) * 0.35)
            + (frame["iso_score"].gt(0).astype(float) * 0.25),
            0,
            1,
        )
        metrics = {
            "theft_auc": None,
            "theft_average_precision": None,
            "theft_precision": None,
            "theft_recall": None,
            "theft_f1": None,
            "anomaly_mode": "unsupervised operational anomaly scoring on unlabelled smart-meter data",
            "visible_false_positive_proxy": "No labels in the operational stream; false positives must be measured after inspection feedback.",
        }
    health = frame.assign(baseline_ok=frame["prev_45"] > 0)
    health["system_drop_ratio"] = health["daily_kwh"] / health["prev_45"].replace(0, np.nan)
    n_unique_meters = health["meter_id"].nunique()
    min_meters_for_valid = max(2, n_unique_meters // 3)
    valid_dates = (
        health[health["baseline_ok"]]
        .groupby("date")
        .agg(median_drop=("system_drop_ratio", "median"), meters=("meter_id", "count"))
        .query(f"meters >= {min_meters_for_valid} and median_drop > 0.15")
        .index
    )
    snapshot_date = max(valid_dates) if len(valid_dates) else frame["date"].max()
    snapshot = frame[frame["date"] <= snapshot_date].sort_values(["meter_id", "date"]).groupby("meter_id").tail(7)
    latest = (
        snapshot.groupby(["meter_id", "feeder_id", "locality", "segment"], as_index=False)
        .agg(
            date=("date", "max"),
            daily_kwh=("daily_kwh", "mean"),
            prev_14=("prev_14", "mean"),
            prev_45=("prev_45", "mean"),
            peer_ratio=("peer_ratio", "mean"),
            drop_ratio=("drop_ratio", "mean"),
            volatility=("volatility", "mean"),
            label_theft=("label_theft", "max"),
            event_type=("event_type", "last"),
            iso_score=("iso_score", "mean"),
            theft_probability=("theft_probability", "mean"),
        )
    )
    latest["rule_score"] = (
        (latest["drop_ratio"] < 0.58).astype(int) * 35
        + (latest["peer_ratio"] < 0.55).astype(int) * 30
        + (latest["iso_score"] > 0).astype(int) * 15
        + (latest["volatility"] > latest["volatility"].quantile(0.8)).astype(int) * 10
    )
    latest["confidence_score"] = np.clip(latest["theft_probability"] * 70 + latest["rule_score"], 0, 100)
    latest = latest[latest["confidence_score"] >= 45].sort_values("confidence_score", ascending=False).head(40)
    latest["confidence"] = pd.cut(
        latest["confidence_score"],
        bins=[0, 60, 78, 101],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)
    latest["estimated_revenue_risk_inr"] = np.round((1 - latest["drop_ratio"].clip(0, 1)) * latest["prev_45"] * 30 * 8.2, 0)
    latest["explanation"] = latest.apply(_explain_anomaly, axis=1)

    out = latest[
        [
            "meter_id",
            "feeder_id",
            "locality",
            "segment",
            "confidence",
            "confidence_score",
            "estimated_revenue_risk_inr",
            "daily_kwh",
            "prev_45",
            "peer_ratio",
            "drop_ratio",
            "event_type",
            "explanation",
        ]
    ].copy()
    return out, metrics


def _explain_anomaly(row: pd.Series) -> str:
    drop_pct = max(0, 100 * (1 - row["drop_ratio"]))
    peer_pct = max(0, 100 * (1 - row["peer_ratio"]))
    action = "physical inspection" if row["confidence"] == "High" else "monitoring and targeted review"
    return (
        f"Meter {row['meter_id']} shows a {drop_pct:.0f}% drop versus its 45-day baseline and "
        f"is {peer_pct:.0f}% below similar {row['segment']} peers in {row['locality']}. "
        f"Confidence is {row['confidence']}; recommended action: {action}."
    )


def build_zone_summary(forecasts: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    risk = (
        forecasts.groupby(["feeder_id", "locality", "capacity_kw"], as_index=False)
        .agg(peak_forecast_kw=("forecast_kw", "max"), max_risk_score=("risk_score", "max"), critical_windows=("risk_level", lambda s: int((s == "critical").sum())))
    )
    anom = anomalies.groupby("feeder_id", as_index=False).agg(open_anomalies=("meter_id", "count"), revenue_risk_inr=("estimated_revenue_risk_inr", "sum"))
    summary = risk.merge(anom, on="feeder_id", how="left").fillna({"open_anomalies": 0, "revenue_risk_inr": 0})
    summary["zone_priority"] = np.where(
        (summary["max_risk_score"] >= 92) | (summary["open_anomalies"] >= 4),
        "Critical",
        np.where((summary["max_risk_score"] >= 78) | (summary["open_anomalies"] >= 2), "High", "Normal"),
    )
    return summary.sort_values(["zone_priority", "max_risk_score"], ascending=[True, False])


def build_anomaly_evidence(df: pd.DataFrame, anomalies: pd.DataFrame) -> dict:
    daily = (
        df.assign(date=pd.to_datetime(df["timestamp"]).dt.date)
        .groupby(["meter_id", "feeder_id", "locality", "segment", "date"], as_index=False)
        .agg(daily_kwh=("kwh", "sum"))
        .sort_values(["meter_id", "date"])
    )
    peer_daily = (
        daily.groupby(["feeder_id", "date"], as_index=False)
        .agg(peer_median_kwh=("daily_kwh", "median"))
    )
    daily = daily.merge(peer_daily, on=["feeder_id", "date"], how="left")
    latest_ts = pd.to_datetime(df["timestamp"]).max()
    recent_raw = df[pd.to_datetime(df["timestamp"]) >= latest_ts - pd.Timedelta(hours=24)].copy()
    evidence: dict[str, dict] = {}
    for anomaly in anomalies.to_dict("records"):
        meter_id = anomaly["meter_id"]
        series = daily[daily["meter_id"] == meter_id].tail(60)
        raw_series = recent_raw[recent_raw["meter_id"] == meter_id].sort_values("timestamp").tail(160)
        evidence[meter_id] = {
            "meter": anomaly,
            "daily_series": [
                {
                    "date": str(row.date),
                    "daily_kwh": round(float(row.daily_kwh), 3),
                    "peer_median_kwh": round(float(row.peer_median_kwh), 3),
                }
                for row in series.itertuples(index=False)
            ],
            "recent_readings": [
                {
                    "timestamp": pd.Timestamp(row.timestamp).isoformat(),
                    "kwh": round(float(row.kwh), 4),
                    "temperature_c": round(float(row.temperature_c), 2),
                }
                for row in raw_series.itertuples(index=False)
            ],
            "decision_rules": [
                {
                    "rule": "Own baseline drop",
                    "value": round(float(1 - anomaly["drop_ratio"]) * 100, 1),
                    "unit": "% below 45-day baseline",
                    "triggered": bool(anomaly["drop_ratio"] < 0.70),
                },
                {
                    "rule": "Peer deviation",
                    "value": round(float(1 - anomaly["peer_ratio"]) * 100, 1),
                    "unit": "% below peer median",
                    "triggered": bool(anomaly["peer_ratio"] < 0.75),
                },
                {
                    "rule": "Revenue exposure",
                    "value": round(float(anomaly["estimated_revenue_risk_inr"]), 0),
                    "unit": "INR/month estimate",
                    "triggered": bool(anomaly["estimated_revenue_risk_inr"] > 0),
                },
            ],
        }
    return evidence


def build_pipeline_summary(metrics: dict) -> dict:
    return {
        "stages": [
            {"name": "Ingest", "status": "complete", "detail": metrics["dataset_source"]},
            {"name": "Clean", "status": "complete", "detail": f"{metrics['meter_rows']:,} rows, {metrics['data_granularity_minutes']}-minute granularity"},
            {"name": "Feature Engineering", "status": "complete", "detail": "Calendar, lag load, rolling load, peer groups, baseline ratios"},
            {"name": "Forecast Model", "status": "complete", "detail": f"LightGBM, MAE {metrics.get('forecast_mae_kw', 0):.3f} kW, sMAPE {metrics.get('forecast_smape', 0) * 100:.1f}%"},
            {"name": "Anomaly Engine", "status": "complete", "detail": metrics.get("anomaly_mode", "Operational scoring")},
            {"name": "Decision Layer", "status": "complete", "detail": "Risk zones, inspection queue, explanations, audit-ready JSON"},
        ],
        "model_cards": [
            {
                "name": "Feeder Demand Forecast",
                "model": "LightGBMRegressor",
                "target": "Next 24-hour feeder load",
                "features": ["hour", "day of week", "real weather", "humidity", "1-hour lag", "24-hour lag", "24-hour rolling mean"],
                "baseline": "24-hour persistence baseline",
                "output": "forecast_kw, lower_kw, upper_kw, risk_score",
            },
            {
                "name": "Meter Anomaly Scoring",
                "model": "IsolationForest + rule layer",
                "target": "Suspicious consumption behaviour",
                "features": ["daily kWh", "14-day baseline", "45-day baseline", "peer ratio", "drop ratio", "volatility"],
                "baseline": "Own-history and peer-group deviation",
                "output": "confidence tier, revenue risk, explanation, inspection recommendation",
            },
            {
                "name": "Labelled Theft Validation",
                "model": "GPU temporal deep net on SGCC",
                "target": "Known theft/non-theft customer label",
                "features": ["daily usage sequence", "first differences", "weekly detrended signal", "missing mask"],
                "baseline": "Class-imbalanced long-sequence binary classification",
                "output": "PR-AUC, ROC-AUC, precision, recall, F1, top suspicious labelled cases",
            },
        ],
    }


def _safe_mean(arr: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nan_to_num(np.nanmean(arr, axis=axis), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_std(arr: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nan_to_num(np.nanstd(arr, axis=axis), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_reduce(fn, arr: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nan_to_num(fn(arr, axis=axis), nan=0.0, posinf=0.0, neginf=0.0)


def _row_linear_slope(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] <= 1:
        return np.zeros(arr.shape[0], dtype=float)
    x = np.arange(arr.shape[1], dtype=float)
    x = x - x.mean()
    denom = float(np.sum(x**2))
    centered = arr - arr.mean(axis=1, keepdims=True)
    return np.nan_to_num(centered @ x / max(denom, 1e-9), nan=0.0, posinf=0.0, neginf=0.0)


def _max_zero_run(mask: np.ndarray) -> np.ndarray:
    runs: list[int] = []
    for row in mask:
        if not row.any():
            runs.append(0)
            continue
        changes = np.diff(np.concatenate(([0], row.astype(int), [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        runs.append(int((ends - starts).max()) if len(starts) else 0)
    return np.asarray(runs, dtype=float)


def _autocorr_feature(arr: np.ndarray, lag: int) -> np.ndarray:
    if arr.shape[1] <= lag:
        return np.zeros(arr.shape[0], dtype=float)
    left = arr[:, :-lag]
    right = arr[:, lag:]
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numer = np.sum(left_centered * right_centered, axis=1)
    denom = np.sqrt(np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1))
    return np.nan_to_num(numer / np.maximum(denom, 1e-9), nan=0.0, posinf=0.0, neginf=0.0)


def _prototype_margin(
    train_frame: pd.DataFrame,
    train_labels: pd.Series,
    eval_frame: pd.DataFrame,
) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = np.nan_to_num(scaler.fit_transform(train_frame), nan=0.0, posinf=0.0, neginf=0.0)
    eval_scaled = np.nan_to_num(scaler.transform(eval_frame), nan=0.0, posinf=0.0, neginf=0.0)
    pos_mask = train_labels.to_numpy() == 1
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return np.zeros(len(eval_frame), dtype=float)
    pos_proto = train_scaled[pos_mask].mean(axis=0)
    neg_proto = train_scaled[neg_mask].mean(axis=0)
    dist_pos = np.linalg.norm(eval_scaled - pos_proto, axis=1)
    dist_neg = np.linalg.norm(eval_scaled - neg_proto, axis=1)
    return np.nan_to_num((dist_neg - dist_pos) / np.maximum(dist_neg + dist_pos, 1e-9), nan=0.0)


def _extract_sgcc_features(values: np.ndarray, ordered_dates: pd.DatetimeIndex) -> pd.DataFrame:
    positive = np.clip(values, 0, None)
    filled = np.nan_to_num(positive, nan=0.0)
    features = pd.DataFrame(
        {
            "mean": _safe_mean(positive, axis=1),
            "std": _safe_std(positive, axis=1),
            "median": _safe_reduce(np.nanmedian, positive, axis=1),
            "max": _safe_reduce(np.nanmax, positive, axis=1),
            "min": _safe_reduce(np.nanmin, positive, axis=1),
            "q10": _safe_reduce(lambda arr, axis: np.nanquantile(arr, 0.10, axis=axis), positive, axis=1),
            "q25": _safe_reduce(lambda arr, axis: np.nanquantile(arr, 0.25, axis=axis), positive, axis=1),
            "q75": _safe_reduce(lambda arr, axis: np.nanquantile(arr, 0.75, axis=axis), positive, axis=1),
            "q90": _safe_reduce(lambda arr, axis: np.nanquantile(arr, 0.90, axis=axis), positive, axis=1),
            "missing_rate": np.isnan(values).mean(axis=1),
            "zero_rate": np.nan_to_num((positive <= 0.001).mean(axis=1)),
        }
    )
    features["range"] = features["max"] - features["min"]
    features["iqr"] = features["q75"] - features["q25"]
    features["cv"] = features["std"] / np.maximum(features["mean"], 0.01)
    features["load_factor"] = features["mean"] / np.maximum(features["max"], 0.01)
    features["peak_to_avg_ratio"] = features["max"] / np.maximum(features["mean"], 0.01)
    features["stability"] = 1 / np.maximum(features["cv"], 0.01)

    recent_30 = _safe_mean(positive[:, -30:], axis=1)
    prev_180 = _safe_mean(positive[:, -210:-30], axis=1)
    first_half = _safe_mean(positive[:, : values.shape[1] // 2], axis=1)
    second_half = _safe_mean(positive[:, values.shape[1] // 2 :], axis=1)
    features["recent_30_mean"] = recent_30
    features["prev_180_mean"] = prev_180
    features["recent_drop_ratio"] = recent_30 / np.maximum(prev_180, 0.01)
    features["half_trend_ratio"] = second_half / np.maximum(first_half, 0.01)

    daily_diffs = np.diff(filled, axis=1)
    features["diff_mean"] = np.abs(daily_diffs).mean(axis=1)
    features["diff_std"] = daily_diffs.std(axis=1)
    features["zero_crossing_rate"] = np.sum(np.diff(np.sign(daily_diffs), axis=1) != 0, axis=1) / max(daily_diffs.shape[1], 1)
    features["max_zero_run"] = _max_zero_run(filled <= 0.001)

    standardized = (filled - features["mean"].to_numpy()[:, None]) / np.maximum(features["std"].to_numpy()[:, None], 0.01)
    features["skewness"] = np.nan_to_num(np.mean(standardized**3, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    features["kurtosis_stat"] = np.nan_to_num(np.mean(standardized**4, axis=1) - 3, nan=0.0, posinf=0.0, neginf=0.0)

    hist_counts = np.apply_along_axis(
        lambda x: np.histogram(x[x > 0], bins=20)[0] if (x > 0).sum() > 5 else np.ones(20),
        1,
        filled,
    )
    hist_probs = hist_counts / np.maximum(hist_counts.sum(axis=1, keepdims=True), 1)
    features["entropy"] = -np.sum(hist_probs * np.log(np.maximum(hist_probs, 1e-12)), axis=1)

    periods = pd.Series(ordered_dates).dt.to_period("M").astype(str).to_numpy()
    monthly_means = []
    for period in sorted(set(periods)):
        mask = periods == period
        month_mean = _safe_mean(positive[:, mask], axis=1)
        monthly_means.append(month_mean)
        features[f"mean_{period}"] = month_mean
        features[f"missing_{period}"] = np.isnan(values[:, mask]).mean(axis=1)
    month_matrix = np.vstack(monthly_means).T if monthly_means else np.zeros((len(values), 1))
    centered_months = month_matrix - month_matrix.mean(axis=1, keepdims=True)
    features["cusum_max"] = np.abs(np.cumsum(centered_months, axis=1)).max(axis=1)
    features["month_std"] = month_matrix.std(axis=1)
    features["month_cv"] = features["month_std"] / np.maximum(month_matrix.mean(axis=1), 0.01)
    features["month_minmax_ratio"] = month_matrix.min(axis=1) / np.maximum(month_matrix.max(axis=1), 0.01)

    full_weeks = filled.shape[1] // 7
    if full_weeks >= 2:
        weekly = filled[:, : full_weeks * 7].reshape(filled.shape[0], full_weeks, 7)
        weekly_means = weekly.mean(axis=2)
        features["weekly_mean_std"] = weekly_means.std(axis=1)
        features["weekly_mean_cv"] = weekly_means.std(axis=1) / np.maximum(weekly_means.mean(axis=1), 0.01)
        features["weekly_slope"] = _row_linear_slope(weekly_means)
    else:
        features["weekly_mean_std"] = 0.0
        features["weekly_mean_cv"] = 0.0
        features["weekly_slope"] = 0.0

    for lag in [7, 14, 30, 60, 90]:
        features[f"autocorr_{lag}"] = _autocorr_feature(filled, lag)

    fft_vals = np.abs(np.fft.rfft(filled, axis=1))[:, 1:]
    fft_power = fft_vals**2
    total_fft_power = np.maximum(fft_power.sum(axis=1), 1e-9)
    bands = {
        "low": slice(0, min(8, fft_power.shape[1])),
        "mid": slice(min(8, fft_power.shape[1]), min(32, fft_power.shape[1])),
        "high": slice(min(32, fft_power.shape[1]), min(128, fft_power.shape[1])),
    }
    for band, band_slice in bands.items():
        band_power = fft_power[:, band_slice].sum(axis=1) if band_slice.start < band_slice.stop else np.zeros(len(values))
        features[f"fft_power_{band}"] = band_power / total_fft_power
    features["fft_max"] = fft_vals.max(axis=1) if fft_vals.shape[1] else 0.0
    features["fft_std"] = fft_vals.std(axis=1) if fft_vals.shape[1] else 0.0

    day_flags = pd.Series(ordered_dates)
    weekend_mask = (day_flags.dt.dayofweek >= 5).to_numpy()
    weekday_mask = ~weekend_mask
    if weekend_mask.any() and weekday_mask.any():
        weekend_mean = _safe_mean(positive[:, weekend_mask], axis=1)
        weekday_mean = _safe_mean(positive[:, weekday_mask], axis=1)
        features["weekend_mean"] = weekend_mean
        features["weekday_mean"] = weekday_mean
        features["weekend_weekday_ratio"] = weekend_mean / np.maximum(weekday_mean, 0.01)

    for window in [7, 14, 30, 60, 90, 180]:
        if filled.shape[1] < window:
            continue
        recent = filled[:, -window:]
        previous = filled[:, -2 * window : -window] if filled.shape[1] >= 2 * window else filled[:, :window]
        features[f"recent_mean_{window}d"] = recent.mean(axis=1)
        features[f"recent_std_{window}d"] = recent.std(axis=1)
        features[f"recent_zero_rate_{window}d"] = (recent <= 0.001).mean(axis=1)
        features[f"recent_slope_{window}d"] = _row_linear_slope(recent)
        features[f"drop_ratio_{window}d"] = recent.mean(axis=1) / np.maximum(previous.mean(axis=1), 0.01)
        features[f"volatility_ratio_{window}d"] = recent.std(axis=1) / np.maximum(previous.std(axis=1), 0.01)
        features[f"min_ratio_{window}d"] = recent.min(axis=1) / np.maximum(previous.mean(axis=1), 0.01)

    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def _make_sgcc_base_models(scale_pos_weight: float, smote_target_ratio: float = 0.0) -> dict[str, object]:
    # If SMOTE rebalances to target_pos_ratio, soften the per-row weighting
    # so positives aren't double-counted by both oversampling and class weight.
    if smote_target_ratio > 0:
        effective_pos_weight = max(1.0, scale_pos_weight * (1.0 - smote_target_ratio))
        et_class_weight = "balanced"
        hist_class_weight = "balanced"
        cat_auto_weights = "Balanced"
    else:
        effective_pos_weight = scale_pos_weight
        et_class_weight = "balanced_subsample"
        hist_class_weight = "balanced"
        cat_auto_weights = "Balanced"
    models: dict[str, object] = {
        "LightGBM": LGBMClassifier(
            n_estimators=2500,
            learning_rate=0.02,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.80,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=effective_pos_weight,
            importance_type="gain",
            random_state=25,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=2500,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.3,
            reg_lambda=3.0,
            min_child_weight=3,
            gamma=0.05,
            max_delta_step=2,
            scale_pos_weight=effective_pos_weight,
            eval_metric="aucpr",
            random_state=26,
            n_jobs=-1,
            tree_method="hist",
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=900,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=et_class_weight,
            random_state=27,
            n_jobs=-1,
        ),
        "HistGB": HistGradientBoostingClassifier(
            max_iter=900,
            learning_rate=0.04,
            max_depth=8,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight=hist_class_weight,
            random_state=29,
        ),
    }
    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=2500,
            depth=7,
            learning_rate=0.025,
            l2_leaf_reg=4.0,
            auto_class_weights=cat_auto_weights,
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=28,
            verbose=False,
            thread_count=-1,
        )
    except ImportError:
        pass
    return models


def _normalized_feature_importance(models: dict[str, object], feature_names: list[str]) -> list[dict]:
    importances = []
    for model in models.values():
        raw = getattr(model, "feature_importances_", None)
        if raw is None:
            continue
        raw = np.asarray(raw, dtype=float)
        if raw.shape[0] != len(feature_names):
            continue
        total = raw.sum()
        if total <= 0:
            continue
        importances.append(raw / total)
    if not importances:
        return [{"feature": feature, "importance": 0.0} for feature in feature_names[:15]]
    mean_importance = np.mean(importances, axis=0)
    return [
        {"feature": feature, "importance": round(float(score), 4)}
        for feature, score in sorted(zip(feature_names, mean_importance), key=lambda item: item[1], reverse=True)[:15]
    ]


def _best_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float, float, float]:
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, score)
    f1_curve = 2 * precision_curve * recall_curve / np.maximum(precision_curve + recall_curve, 1e-9)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0
    best_idx = int(np.nanargmax(f1_curve[:-1]))
    return (
        float(thresholds[best_idx]),
        float(f1_curve[best_idx]),
        float(precision_curve[best_idx]),
        float(recall_curve[best_idx]),
    )


def _resample_with_smote(
    X_fit: pd.DataFrame, y_fit: pd.Series, target_pos_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    if not HAS_IMBLEARN:
        return X_fit, y_fit
    pos = int((y_fit == 1).sum())
    neg = int((y_fit == 0).sum())
    if pos == 0 or neg == 0:
        return X_fit, y_fit
    target_pos = int(neg * target_pos_ratio / max(1.0 - target_pos_ratio, 1e-3))
    if target_pos <= pos:
        return X_fit, y_fit
    sm = BorderlineSMOTE(
        sampling_strategy={1: target_pos},
        k_neighbors=min(5, max(1, pos - 1)),
        random_state=seed,
    )
    X_res, y_res = sm.fit_resample(X_fit, y_fit)
    return X_res, y_res


def _fit_with_early_stopping(
    name: str,
    model: object,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    seed: int,
) -> object:
    fold_model = clone(model)
    if name in {"LightGBM", "XGBoost", "CatBoost"}:
        X_fit_sub, X_es, y_fit_sub, y_es = train_test_split(
            X_fit, y_fit, test_size=0.15, stratify=y_fit, random_state=seed
        )
        if name == "LightGBM":
            fold_model.fit(
                X_fit_sub, y_fit_sub,
                eval_set=[(X_es, y_es)],
                callbacks=[lgbm.early_stopping(stopping_rounds=120, verbose=False)],
            )
        elif name == "XGBoost":
            fold_model.set_params(early_stopping_rounds=120)
            fold_model.fit(X_fit_sub, y_fit_sub, eval_set=[(X_es, y_es)], verbose=False)
        else:  # CatBoost
            fold_model.fit(
                X_fit_sub, y_fit_sub,
                eval_set=(X_es, y_es),
                early_stopping_rounds=120,
                verbose=False,
            )
    else:
        fold_model.fit(X_fit, y_fit)
    return fold_model


def _fit_oof(
    base_models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cv: StratifiedKFold,
    smote_target_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    oof = pd.DataFrame(index=X_train.index, columns=list(base_models.keys()), dtype=float)
    test = pd.DataFrame(index=X_test.index, columns=list(base_models.keys()), dtype=float)
    fitted: dict[str, object] = {}
    smote_active = HAS_IMBLEARN and smote_target_ratio > 0
    print(f"  SMOTE: {'BorderlineSMOTE target_pos_ratio=' + str(smote_target_ratio) if smote_active else 'disabled'}")
    for name, model in base_models.items():
        oof_pred = np.zeros(len(X_train), dtype=float)
        test_pred_folds = np.zeros(len(X_test), dtype=float)
        for fold_idx, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
            X_fit = X_train.iloc[fit_idx]
            y_fit = y_train.iloc[fit_idx]
            X_val = X_train.iloc[val_idx]
            if smote_active:
                X_fit, y_fit = _resample_with_smote(
                    X_fit, y_fit, smote_target_ratio, seed=1000 * fold_idx + hash(name) % 997
                )
            fold_model = _fit_with_early_stopping(name, model, X_fit, y_fit, seed=fold_idx)
            oof_pred[val_idx] = fold_model.predict_proba(X_val)[:, 1]
            test_pred_folds += fold_model.predict_proba(X_test)[:, 1] / cv.get_n_splits()
            print(f"  {name:14s} fold {fold_idx}/{cv.get_n_splits()} done")
        if smote_active:
            X_full, y_full = _resample_with_smote(X_train, y_train, smote_target_ratio, seed=99)
        else:
            X_full, y_full = X_train, y_train
        full_model = _fit_with_early_stopping(name, model, X_full, y_full, seed=99)
        fitted[name] = full_model
        oof[name] = oof_pred
        test[name] = 0.5 * test_pred_folds + 0.5 * full_model.predict_proba(X_test)[:, 1]
        print(
            f"  {name:14s} OOF AP={average_precision_score(y_train, oof_pred):.4f} "
            f"OOF AUC={roc_auc_score(y_train, oof_pred):.4f}"
        )
    return oof, test, fitted


def build_sgcc_theft_validation(path: Path = SGCC_CSV) -> dict:
    if not path.exists():
        return {
            "available": False,
            "reason": "SGCC split archive has not been extracted yet.",
        }
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

    print(f"SGCC: {len(raw)} customers, {len(ordered_cols)} days, positive rate {labels.mean():.4f}")

    features = _extract_sgcc_features(values, ordered_dates)
    variance_selector = VarianceThreshold(threshold=1e-5)
    reduced = pd.DataFrame(
        variance_selector.fit_transform(features),
        columns=features.columns[variance_selector.get_support()],
        index=features.index,
    )
    kbest = SelectKBest(f_classif, k=min(120, len(reduced.columns)))
    selected_array = kbest.fit_transform(reduced, labels)
    selected_feature_names = reduced.columns[kbest.get_support()].tolist()
    features_selected = pd.DataFrame(selected_array, columns=selected_feature_names, index=features.index)
    print(f"Feature selection: {len(features.columns)} -> {len(features_selected.columns)} features")

    all_idx = np.arange(len(labels))
    train_idx_arr, test_idx_arr = train_test_split(
        all_idx, test_size=0.20, stratify=labels, random_state=23
    )
    X_train = features_selected.iloc[train_idx_arr].reset_index(drop=True)
    X_test = features_selected.iloc[test_idx_arr].reset_index(drop=True)
    y_train = labels.iloc[train_idx_arr].reset_index(drop=True)
    y_test = labels.iloc[test_idx_arr].reset_index(drop=True)
    id_train = raw["CONS_NO"].iloc[train_idx_arr].reset_index(drop=True)
    id_test = raw["CONS_NO"].iloc[test_idx_arr].reset_index(drop=True)
    class_counts = np.bincount(y_train)
    scale_pos_weight = float(class_counts[0] / max(class_counts[1], 1))
    print(
        f"Train class counts: neg={class_counts[0]} pos={class_counts[1]} "
        f"scale_pos_weight={scale_pos_weight:.2f}"
    )

    smote_target_ratio = 0.20 if HAS_IMBLEARN else 0.0
    base_models = _make_sgcc_base_models(scale_pos_weight, smote_target_ratio=smote_target_ratio)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=== OOF training of base models ===")
    oof_base, test_base, fitted_models = _fit_oof(
        base_models, X_train, y_train, X_test, cv, smote_target_ratio=smote_target_ratio
    )

    deep_mode = os.environ.get("SGCC_DEEP", "lite")
    if HAS_TORCH and deep_mode != "off":
        print(f"\n=== Deep base model (CNN only, mode={deep_mode}) ===")
        cnn_inputs = prepare_cnn_inputs(values)
        aux_array = features_selected.to_numpy(dtype=np.float32)
        aux_mean = aux_array[train_idx_arr].mean(axis=0, keepdims=True)
        aux_std = np.maximum(aux_array[train_idx_arr].std(axis=0, keepdims=True), 1e-3)
        aux_array = (aux_array - aux_mean) / aux_std
        n_aux = aux_array.shape[1]
        y_arr = labels.to_numpy(dtype=np.int64)

        if deep_mode == "full":
            deep_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            deep_epochs, deep_batch = 8, 256
        else:  # 'lite'
            deep_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            deep_epochs, deep_batch = 4, 512

        cnn_oof, cnn_test = train_oof_deep(
            lambda: SGCCConvNet(in_channels=cnn_inputs.shape[1], n_aux=n_aux),
            cnn_inputs, aux_array, y_arr, train_idx_arr, test_idx_arr, deep_cv,
            name="CNN", max_epochs=deep_epochs, patience=2, batch_size=deep_batch,
        )
        oof_base["CNN"] = cnn_oof
        test_base["CNN"] = cnn_test
    else:
        print(f"\n[skipping deep base models: torch={HAS_TORCH} mode={deep_mode}]")

    print("\n=== OOF prototype margin and anomaly meta-features ===")
    proto_oof = np.zeros(len(X_train), dtype=float)
    for fit_idx, val_idx in cv.split(X_train, y_train):
        proto_oof[val_idx] = _prototype_margin(
            X_train.iloc[fit_idx], y_train.iloc[fit_idx], X_train.iloc[val_idx]
        )
    proto_test = _prototype_margin(X_train, y_train, X_test)
    oof_base["prototype_margin"] = proto_oof
    test_base["prototype_margin"] = proto_test
    oof_base["normal_distance_score"] = 1 - proto_oof
    test_base["normal_distance_score"] = 1 - proto_test

    blend_columns = list(base_models.keys())
    if "CNN" in oof_base.columns:
        blend_columns += ["CNN"]
    blend_train_matrix = oof_base[blend_columns].to_numpy()
    blend_test_matrix = test_base[blend_columns].to_numpy()
    print(f"Blend columns: {blend_columns}")

    def _blend_objective(weights: np.ndarray) -> float:
        weights = np.abs(weights)
        weights = weights / np.maximum(weights.sum(), 1e-9)
        blend_score = blend_train_matrix @ weights
        return -average_precision_score(y_train.to_numpy(), blend_score)

    blend_result = differential_evolution(
        _blend_objective,
        bounds=[(0.05, 1.0)] * len(blend_columns),
        seed=45,
        maxiter=60,
        popsize=15,
        polish=True,
    )
    blend_weights = np.abs(blend_result.x)
    blend_weights = blend_weights / np.maximum(blend_weights.sum(), 1e-9)
    print(f"Blend weights: {dict(zip(blend_columns, np.round(blend_weights, 4)))}")
    blend_train_raw = blend_train_matrix @ blend_weights
    blend_test_raw = blend_test_matrix @ blend_weights

    print("\n=== Pseudo-labeling: ensemble label-noise correction ===")
    ensemble_consensus = oof_base[blend_columns].mean(axis=1).to_numpy()
    y_train_arr = y_train.to_numpy()
    suspicious_pos_to_neg = (y_train_arr == 1) & (ensemble_consensus < 0.05)
    suspicious_neg_to_pos = (y_train_arr == 0) & (ensemble_consensus > 0.95)
    suspicious_mask = suspicious_pos_to_neg | suspicious_neg_to_pos
    n_susp = int(suspicious_mask.sum())
    print(
        f"Flagged {n_susp} suspicious labels ({suspicious_pos_to_neg.sum()} pos→neg, "
        f"{suspicious_neg_to_pos.sum()} neg→pos) — dropping from meta-learner training only"
    )
    keep_mask = ~suspicious_mask
    keep_idx = np.where(keep_mask)[0]

    print("\n=== Out-of-fold meta-learner stacking ===")
    meta_features = oof_base.copy()
    meta_features_test = test_base.copy()
    meta_oof = np.zeros(len(X_train), dtype=float)
    meta_test_folds = np.zeros(len(X_test), dtype=float)
    cv_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
    meta_features_clean = meta_features.iloc[keep_idx]
    y_train_clean = y_train.iloc[keep_idx]
    for fit_local, val_local in cv_meta.split(meta_features_clean, y_train_clean):
        fit_global = keep_idx[fit_local]
        val_global = keep_idx[val_local]
        meta_fold = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=4000,
            solver="lbfgs",
            random_state=44,
        )
        meta_fold.fit(meta_features.iloc[fit_global], y_train.iloc[fit_global])
        meta_oof[val_global] = meta_fold.predict_proba(meta_features.iloc[val_global])[:, 1]
        meta_test_folds += meta_fold.predict_proba(meta_features_test)[:, 1] / cv_meta.get_n_splits()
    # Suspicious rows still need an OOF prediction for fair F1 reporting
    if n_susp > 0:
        meta_susp_pred = LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=4000, solver="lbfgs", random_state=44,
        )
        meta_susp_pred.fit(meta_features.iloc[keep_idx], y_train.iloc[keep_idx])
        meta_oof[suspicious_mask] = meta_susp_pred.predict_proba(
            meta_features.iloc[suspicious_mask]
        )[:, 1]

    meta_full = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=4000,
        solver="lbfgs",
        random_state=44,
    )
    meta_full.fit(meta_features.iloc[keep_idx], y_train.iloc[keep_idx])
    meta_test_raw = 0.5 * meta_test_folds + 0.5 * meta_full.predict_proba(meta_features_test)[:, 1]

    meta_calibrator = IsotonicRegression(out_of_bounds="clip")
    meta_oof_cal = meta_calibrator.fit_transform(meta_oof, y_train)
    meta_test_cal = meta_calibrator.transform(meta_test_raw)

    blend_calibrator = IsotonicRegression(out_of_bounds="clip")
    blend_oof_cal = blend_calibrator.fit_transform(blend_train_raw, y_train)
    blend_test_cal = blend_calibrator.transform(blend_test_raw)

    meta_threshold, meta_oof_f1, meta_oof_p, meta_oof_r = _best_f1_threshold(y_train.to_numpy(), meta_oof_cal)
    blend_threshold, blend_oof_f1, blend_oof_p, blend_oof_r = _best_f1_threshold(y_train.to_numpy(), blend_oof_cal)

    print(f"OOF meta   : F1={meta_oof_f1:.4f}  P={meta_oof_p:.4f}  R={meta_oof_r:.4f}  thr={meta_threshold:.4f}")
    print(f"OOF blend  : F1={blend_oof_f1:.4f}  P={blend_oof_p:.4f}  R={blend_oof_r:.4f}  thr={blend_threshold:.4f}")

    if blend_oof_f1 > meta_oof_f1:
        proba = blend_test_cal
        threshold = blend_threshold
        selected_mode = f"weighted_blend {dict(zip(blend_columns, np.round(blend_weights, 4)))}"
        oof_proba = blend_oof_cal
    else:
        proba = meta_test_cal
        threshold = meta_threshold
        selected_mode = "meta_logistic_stack"
        oof_proba = meta_oof_cal
    print(f"Selected: {selected_mode}  threshold(OOF)={threshold:.4f}")

    pred = (proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, proba)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, proba)

    test_frame = X_test.copy()
    test_frame["recent_drop_ratio"] = features.loc[X_test.index, "recent_drop_ratio"].to_numpy()
    test_frame["missing_rate"] = features.loc[X_test.index, "missing_rate"].to_numpy()
    test_frame["zero_rate"] = features.loc[X_test.index, "zero_rate"].to_numpy()
    test_frame["consumer_id"] = id_test.to_numpy()
    test_frame["label"] = y_test.to_numpy()
    test_frame["theft_probability"] = proba
    top_cases = test_frame.sort_values("theft_probability", ascending=False).head(12)
    cases = []
    for row in top_cases.to_dict("records"):
        drop_pct = max(0, (1 - row["recent_drop_ratio"]) * 100)
        cases.append(
            {
                "consumer_id": row["consumer_id"],
                "label": int(row["label"]),
                "theft_probability": round(float(row["theft_probability"]), 4),
                "recent_drop_pct": round(float(drop_pct), 1),
                "missing_rate": round(float(row["missing_rate"]), 3),
                "zero_rate": round(float(row["zero_rate"]), 3),
                "explanation": (
                    f"SGCC customer {row['consumer_id']} has {drop_pct:.0f}% recent drop vs historical baseline, "
                    f"{row['zero_rate']:.0%} near-zero days, and model probability {row['theft_probability']:.2f}."
                ),
            }
        )

    model_package = {
        "base_models": fitted_models,
        "meta_model": meta_full,
        "meta_calibrator": meta_calibrator,
        "blend_calibrator": blend_calibrator,
        "selected_features": selected_feature_names,
        "stack_features": list(meta_features.columns),
        "selected_mode": selected_mode,
        "blend_weights": dict(zip(blend_columns, [float(x) for x in blend_weights])),
        "threshold": threshold,
    }
    joblib.dump(model_package, MODELS_DIR / "sgcc_theft_stacking.joblib")
    joblib.dump(fitted_models, MODELS_DIR / "sgcc_theft_ensemble.joblib")

    return {
        "available": True,
        "dataset": "SGCC Electricity Theft Detection",
        "customers": int(len(raw)),
        "days": int(len(ordered_cols)),
        "positive_rate": round(float(labels.mean()), 4),
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "oof_f1": round(float(max(meta_oof_f1, blend_oof_f1)), 4),
        "oof_pr_auc": round(float(average_precision_score(y_train, oof_proba)), 4),
        "oof_roc_auc": round(float(roc_auc_score(y_train, oof_proba)), 4),
        "model": (
            "Leakage-free OOF stacked ensemble: tabular trees (LightGBM, XGBoost, ExtraTrees, "
            "HistGradientBoosting, CatBoost) + deep temporal models (1D-CNN, BiGRU on weekly aggregates), "
            "BorderlineSMOTE per fold, prototype-margin and anomaly meta-features, "
            "ensemble label-noise correction, and isotonic-calibrated meta logistic stack."
        ),
        "component_metrics": {
            name: {
                "roc_auc": round(float(roc_auc_score(y_test, test_base[name])), 4),
                "pr_auc": round(float(average_precision_score(y_test, test_base[name])), 4),
            }
            for name in blend_columns
        },
        "confusion_matrix": cm,
        "roc_curve": [
            {"fpr": round(float(x), 4), "tpr": round(float(y), 4)}
            for x, y in zip(fpr[:: max(1, len(fpr) // 80)], tpr[:: max(1, len(tpr) // 80)])
        ],
        "pr_curve": [
            {"recall": round(float(x), 4), "precision": round(float(y), 4)}
            for x, y in zip(pr_recall[:: max(1, len(pr_recall) // 80)], pr_precision[:: max(1, len(pr_precision) // 80)])
        ],
        "feature_importance": _normalized_feature_importance(fitted_models, selected_feature_names),
        "top_cases": cases,
    }



def run_pipeline(source: str = "auto") -> dict:
    ensure_dirs()

    print("=" * 60)
    print("RUNNING SGCC THEFT VALIDATION (LABELLED DATA)")
    print("=" * 60)
    theft_validation = build_sgcc_theft_validation(SGCC_CSV)

    if source == "sgcc" or (source == "auto" and SGCC_CSV.exists()):
        df = generate_synthetic_meter_data()
        dataset_source = "Synthetic operational data + SGCC theft validation"
    elif source == "london" or (source == "auto" and LONDON_HH_BLOCK.exists() and LONDON_WEATHER.exists()):
        df = load_london_real_data()
        dataset_source = "Kaggle London smart-meter dataset with real weather"
    elif source == "india" or (source == "auto" and INDIA_PRIMARY.exists()):
        df = load_india_real_data()
        dataset_source = "Kaggle CEEW Indian smart-meter dataset"
    elif source == "lead" or (source == "auto" and LEAD_ZIP.exists()):
        df = load_lead_real_data()
        dataset_source = "LEAD 1.0 small real smart-meter dataset"
    else:
        df = generate_synthetic_meter_data()
        dataset_source = "synthetic fallback"
        
    forecasts, forecast_metrics = build_forecasts(df)
    anomalies, anomaly_metrics = build_anomalies(df)
    zones = build_zone_summary(forecasts, anomalies)
    metrics = {
        **forecast_metrics,
        **anomaly_metrics,
        "meters": int(df["meter_id"].nunique()),
        "feeders": int(df["feeder_id"].nunique()),
        "meter_rows": int(len(df)),
        "data_granularity_minutes": int(pd.to_datetime(df["timestamp"]).sort_values().drop_duplicates().diff().dt.total_seconds().dropna().mode().iloc[0] / 60),
        "dataset_source": dataset_source,
        "sgcc_theft_validation": theft_validation.get("available", False),
        "sgcc_theft_f1": theft_validation.get("f1"),
        "sgcc_theft_pr_auc": theft_validation.get("pr_auc"),
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
    }

    df.sample(min(5000, len(df)), random_state=1).to_csv(DATA_DIR / "sample_meter_readings.csv", index=False)
    forecasts.to_json(DATA_DIR / "forecasts.json", orient="records", date_format="iso", indent=2)
    anomalies.to_json(DATA_DIR / "anomalies.json", orient="records", indent=2)
    zones.to_json(DATA_DIR / "zones.json", orient="records", indent=2)
    (DATA_DIR / "anomaly_evidence.json").write_text(json.dumps(build_anomaly_evidence(df, anomalies), indent=2), encoding="utf-8")
    (DATA_DIR / "pipeline_summary.json").write_text(json.dumps(build_pipeline_summary(metrics), indent=2), encoding="utf-8")
    (DATA_DIR / "theft_validation.json").write_text(json.dumps(theft_validation, indent=2), encoding="utf-8")
    (DATA_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
