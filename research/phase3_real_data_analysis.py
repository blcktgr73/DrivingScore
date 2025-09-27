"""
Phase 3: Validate driving score scenarios with real Kaggle motion sensor data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
WINDOW_SIZE = 8
DATA_DIR = Path("data/phase3")
EVENTS_FULL = ("rapid_acceleration", "sudden_stop", "sharp_turn", "over_speeding")
ENV_FEATURES = ("night_ratio", "mean_accel_mag", "gyro_abs_mean", "accel_std")

ACCEL_THRESHOLD = 1.2
DECEL_THRESHOLD = -1.2
TURN_THRESHOLD = 1.0


@dataclass
class ModelMetrics:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def load_motion_dataset(data_dir: Path = DATA_DIR, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    train_path = data_dir / "train_motion_data.csv"
    test_path = data_dir / "test_motion_data.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Kaggle motion data not found. Please download to data/phase3.")

    train = pd.read_csv(train_path)
    train["split"] = "train"
    test = pd.read_csv(test_path)
    test["split"] = "test"

    speed_metric_all = np.hypot(np.concatenate([train["AccX"], test["AccX"]]),
                                np.concatenate([train["AccY"], test["AccY"]]))
    overspeed_threshold = float(np.quantile(speed_metric_all, 0.92))

    train_agg = aggregate_motion(train, split="train", window_size=window_size, overspeed_threshold=overspeed_threshold)
    test_agg = aggregate_motion(test, split="test", window_size=window_size, overspeed_threshold=overspeed_threshold)

    combined = pd.concat([train_agg, test_agg], ignore_index=True)
    return combined


def aggregate_motion(df: pd.DataFrame, *, split: str, window_size: int, overspeed_threshold: float) -> pd.DataFrame:
    data = df.sort_values("Timestamp").copy()
    data["is_night"] = ((data["Timestamp"] // 5) % 2).astype(int)
    data["speed_metric_xy"] = np.hypot(data["AccX"], data["AccY"])
    data["window_id"] = (data["Timestamp"] - data["Timestamp"].min()) // window_size

    rows = []
    for window_id, group in data.groupby("window_id"):
        if len(group) < 6:
            continue
        majority_class = group["Class"].value_counts().idxmax()
        label = int(majority_class == "AGGRESSIVE")
        day = group[group["is_night"] == 0]
        night = group[group["is_night"] == 1]

        row = {
            "window_id": int(window_id),
            "split": split,
            "label": label,
            "sample_size": int(len(group)),
            "night_ratio": float(group["is_night"].mean()),
            "rapid_acceleration_day": int((day["AccX"] > ACCEL_THRESHOLD).sum()),
            "rapid_acceleration_night": int((night["AccX"] > ACCEL_THRESHOLD).sum()),
            "sudden_stop_day": int((day["AccX"] < DECEL_THRESHOLD).sum()),
            "sudden_stop_night": int((night["AccX"] < DECEL_THRESHOLD).sum()),
            "sharp_turn_day": int((day["GyroZ"].abs() > TURN_THRESHOLD).sum()),
            "sharp_turn_night": int((night["GyroZ"].abs() > TURN_THRESHOLD).sum()),
            "over_speeding_day": int((day["speed_metric_xy"] > overspeed_threshold).sum()),
            "over_speeding_night": int((night["speed_metric_xy"] > overspeed_threshold).sum()),
            "mean_accel_mag": float(np.sqrt((group[["AccX", "AccY", "AccZ"]] ** 2).sum(axis=1)).mean()),
            "gyro_abs_mean": float(group[["GyroX", "GyroY", "GyroZ"]].abs().mean().mean()),
            "accel_std": float(group[["AccX", "AccY", "AccZ"]].std().mean()),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def prepare_features(df: pd.DataFrame, events: Iterable[str]) -> Tuple[pd.DataFrame, pd.Series]:
    feature_columns = []
    for event in events:
        feature_columns.append(f"{event}_day")
        feature_columns.append(f"{event}_night")
    feature_columns.extend(ENV_FEATURES)

    X = df[feature_columns].copy()
    y = df["label"].astype(int)
    return X, y


def evaluate_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                    logistic_model: LogisticRegression) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}

    lr_prob = logistic_model.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    metrics["logistic_regression"] = _calc_metrics("logistic_regression", y_test, lr_pred, lr_prob).to_dict()

    xgb_model = XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)
    metrics["xgboost"] = _calc_metrics("xgboost", y_test, xgb_pred, xgb_prob).to_dict()

    lgb_model = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgb_model.fit(X_train, y_train)
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = (lgb_prob >= 0.5).astype(int)
    metrics["lightgbm"] = _calc_metrics("lightgbm", y_test, lgb_pred, lgb_prob).to_dict()

    return metrics


def _calc_metrics(name: str, y_true, y_pred, y_prob) -> ModelMetrics:
    return ModelMetrics(
        name=name,
        accuracy=round(accuracy_score(y_true, y_pred), 4),
        precision=round(precision_score(y_true, y_pred, zero_division=0), 4),
        recall=round(recall_score(y_true, y_pred, zero_division=0), 4),
        f1=round(f1_score(y_true, y_pred, zero_division=0), 4),
        auc=round(roc_auc_score(y_true, y_prob), 4),
    )


def derive_event_weights(model: LogisticRegression, feature_names: Iterable[str], events: Iterable[str]) -> Dict[str, Dict[str, float]]:
    coef = pd.Series(model.coef_[0], index=list(feature_names))

    day_raw = {event: abs(coef.get(f"{event}_day", 0.0)) for event in events}
    night_raw = {event: abs(coef.get(f"{event}_night", 0.0)) for event in events}

    day_scale = 2.2 / (np.mean(list(day_raw.values())) + 1e-6)
    night_scale = 3.1 / (np.mean(list(night_raw.values())) + 1e-6)

    day_weights = {event: round(value * day_scale, 2) for event, value in day_raw.items()}
    night_weights = {event: round(value * night_scale, 2) for event, value in night_raw.items()}

    return {
        "day": day_weights,
        "night": night_weights,
        "raw_coefficients": coef.to_dict(),
    }


def derive_environment_params(model: LogisticRegression, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    coef = pd.Series(model.coef_[0], index=model.feature_names_in_)
    env_coeffs = {feat: float(coef.get(feat, 0.0)) for feat in ENV_FEATURES}
    env_means = {feat: float(df[feat].mean()) for feat in ENV_FEATURES}
    return {
        "coeffs": env_coeffs,
        "means": env_means,
        "caps": (0.85, 1.9),
    }


def calculate_scores(df: pd.DataFrame, events: Iterable[str], event_weights: Dict[str, Dict[str, float]],
                     env_params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    scored = df.copy()
    day_weights = event_weights["day"]
    night_weights = event_weights["night"]

    penalties = np.zeros(len(scored))
    for event in events:
        penalties += scored[f"{event}_day"].to_numpy() * day_weights.get(event, 0.0)
        penalties += scored[f"{event}_night"].to_numpy() * night_weights.get(event, 0.0)

    coeffs = env_params["coeffs"]
    means = env_params["means"]
    cap_low, cap_high = env_params["caps"]

    env_component = np.zeros(len(scored))
    for feat, weight in coeffs.items():
        env_component += weight * (scored[feat].to_numpy() - means.get(feat, 0.0))

    env_factor = np.clip(np.exp(env_component), cap_low, cap_high)

    scored["base_penalty"] = penalties
    scored["environment_factor"] = env_factor
    scored["adjusted_penalty"] = penalties * env_factor
    scored["safety_score"] = np.clip(100 - scored["adjusted_penalty"], 0, 100)
    return scored


def tune_grade_thresholds(df: pd.DataFrame) -> Dict[str, object]:
    scores = df["safety_score"].to_numpy()
    labels = df["label"].to_numpy()

    aggressive_candidates = np.arange(50, 81, 0.5)
    safe_candidates = np.arange(65, 91, 0.5)

    best_aggressive, best_aggressive_youden = 68.0, -1.0
    best_safe, best_safe_youden = 78.0, -1.0

    for threshold in aggressive_candidates:
        predictions = (scores <= threshold).astype(int)
        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        tn = int(((predictions == 0) & (labels == 0)).sum())
        youden = _compute_youden(tp, fp, tn, fn)
        if youden > best_aggressive_youden:
            best_aggressive_youden = youden
            best_aggressive = float(threshold)

    inverse_labels = 1 - labels
    for threshold in safe_candidates:
        predictions = (scores >= threshold).astype(int)
        tp = int(((predictions == 1) & (inverse_labels == 1)).sum())
        fp = int(((predictions == 1) & (inverse_labels == 0)).sum())
        fn = int(((predictions == 0) & (inverse_labels == 1)).sum())
        tn = int(((predictions == 0) & (inverse_labels == 0)).sum())
        youden = _compute_youden(tp, fp, tn, fn)
        if youden > best_safe_youden:
            best_safe_youden = youden
            best_safe = float(threshold)

    if best_safe <= best_aggressive + 3:
        best_safe = min(best_aggressive + 6, 95.0)

    bins = [-np.inf, best_aggressive, best_safe, np.inf]
    labels_names = ["AGGRESSIVE", "MODERATE", "SAFE"]
    df["grade"] = pd.cut(df["safety_score"], bins=bins, labels=labels_names, include_lowest=True)

    grade_distribution = df["grade"].value_counts(normalize=True).sort_index().round(4).to_dict()
    grade_accident_rates = df.groupby("grade")["label"].mean().round(4).to_dict()

    return {
        "aggressive_cut": round(best_aggressive, 1),
        "safe_cut": round(best_safe, 1),
        "moderate_range": [round(best_aggressive + 0.1, 1), round(best_safe - 0.1, 1)],
        "grade_distribution": grade_distribution,
        "grade_accident_rate": grade_accident_rates,
    }


def _compute_youden(tp: int, fp: int, tn: int, fn: int) -> float:
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return tpr - fpr


def run_scenario(agg_df: pd.DataFrame, events: Tuple[str, ...], scenario_name: str) -> Dict[str, object]:
    X, y = prepare_features(agg_df, events)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    logistic_model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    logistic_model.fit(X_train, y_train)

    metrics = evaluate_models(X_train, X_test, y_train, y_test, logistic_model)

    logistic_full = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    logistic_full.fit(X, y)
    event_weights = derive_event_weights(logistic_full, X.columns, events)
    env_params = derive_environment_params(logistic_full, agg_df[list(ENV_FEATURES)])

    scored_df = calculate_scores(agg_df, events, event_weights, env_params)
    thresholds = tune_grade_thresholds(scored_df.copy())

    score_stats = scored_df["safety_score"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict()

    return {
        "label": scenario_name,
        "event_weights": {
            "day": event_weights["day"],
            "night": event_weights["night"],
        },
        "environment_params": env_params,
        "model_metrics": metrics,
        "score_summary": {
            "safety_score_stats": score_stats,
            "grade_distribution": thresholds["grade_distribution"],
            "grade_accident_rate": thresholds["grade_accident_rate"],
        },
        "thresholds": {
            "aggressive_cut": thresholds["aggressive_cut"],
            "safe_cut": thresholds["safe_cut"],
            "moderate_range": thresholds["moderate_range"],
        },
        "scored_records": int(len(scored_df)),
    }


def build_comparison(result_with: Dict[str, object], result_without: Dict[str, object]) -> Dict[str, object]:
    comparison: Dict[str, object] = {}

    def model_delta(metric: str, model: str) -> float:
        return round(result_without["model_metrics"][model][metric] - result_with["model_metrics"][model][metric], 4)

    comparison["model_deltas"] = {
        model: {
            metric: model_delta(metric, model)
            for metric in ("auc", "f1", "recall", "precision", "accuracy")
        }
        for model in ("logistic_regression", "xgboost", "lightgbm")
    }

    comparison["grade_distribution_delta"] = {
        grade: round(
            result_without["score_summary"]["grade_distribution"].get(grade, 0.0)
            - result_with["score_summary"]["grade_distribution"].get(grade, 0.0),
            4,
        )
        for grade in ("SAFE", "MODERATE", "AGGRESSIVE")
    }

    comparison["threshold_delta"] = {
        "aggressive_cut": round(result_without["thresholds"]["aggressive_cut"] - result_with["thresholds"]["aggressive_cut"], 2),
        "safe_cut": round(result_without["thresholds"]["safe_cut"] - result_with["thresholds"]["safe_cut"], 2),
        "moderate_range": [
            round(result_without["thresholds"]["moderate_range"][0] - result_with["thresholds"]["moderate_range"][0], 2),
            round(result_without["thresholds"]["moderate_range"][1] - result_with["thresholds"]["moderate_range"][1], 2),
        ],
    }

    return comparison


def main(output_path: Path = Path("research/phase3_results.json")) -> Dict[str, object]:
    aggregated = load_motion_dataset()
    dataset_summary = {
        "records": int(len(aggregated)),
        "positive_rate": round(float(aggregated["label"].mean()), 4),
        "mean_night_ratio": round(float(aggregated["night_ratio"].mean()), 4),
    }

    scenario_with = run_scenario(aggregated, EVENTS_FULL, "Scenario A: with overspeeding")
    scenario_without = run_scenario(aggregated, EVENTS_FULL[:-1], "Scenario B: without overspeeding")

    comparison = build_comparison(scenario_with, scenario_without)

    results = {
        "dataset": dataset_summary,
        "scenario_with": scenario_with,
        "scenario_without": scenario_without,
        "comparison": comparison,
    }

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("Phase 3 (real data) scenarios completed")
    print(f"  Aggregated windows: {dataset_summary['records']} | Aggressive rate: {dataset_summary['positive_rate']:.2%}")
    print("  Scenario A AUC (LR/XGB/LGBM):",
          scenario_with["model_metrics"]["logistic_regression"]["auc"],
          scenario_with["model_metrics"]["xgboost"]["auc"],
          scenario_with["model_metrics"]["lightgbm"]["auc"])
    print("  Scenario B AUC (LR/XGB/LGBM):",
          scenario_without["model_metrics"]["logistic_regression"]["auc"],
          scenario_without["model_metrics"]["xgboost"]["auc"],
          scenario_without["model_metrics"]["lightgbm"]["auc"])

    return results


if __name__ == "__main__":
    main()
