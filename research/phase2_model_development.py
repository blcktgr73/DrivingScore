"""
Phase 2: Compare safety-score models with and without overspeeding events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DAY_SUFFIX = "_day"
NIGHT_SUFFIX = "_night"
BASE_EVENTS = ("rapid_acceleration", "sudden_stop", "sharp_turn", "over_speeding")
EVENT_LABELS = {
    "rapid_acceleration": "rapid_acceleration",
    "sudden_stop": "sudden_stop",
    "sharp_turn": "sharp_turn",
    "over_speeding": "over_speeding",
}

SCENARIOS = {
    "with_overspeed": {
        "label": "시나리오 A: 과속 포함 (4개 이벤트)",
        "events": ("rapid_acceleration", "sudden_stop", "sharp_turn", "over_speeding"),
    },
    "without_overspeed": {
        "label": "시나리오 B: 과속 제외 (3개 이벤트)",
        "events": ("rapid_acceleration", "sudden_stop", "sharp_turn"),
    },
}


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


def generate_phase2_dataset(n_samples: int = 35000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    exposure_index = rng.gamma(shape=2.2, scale=1.2, size=n_samples)
    night_ratio = np.clip(rng.beta(2.0, 5.0, size=n_samples) + rng.normal(0, 0.05, size=n_samples), 0, 0.9)
    weather = rng.choice(["clear", "rain", "storm"], size=n_samples, p=[0.63, 0.28, 0.09])
    road_type = rng.choice(["highway", "urban", "suburban"], size=n_samples, p=[0.35, 0.45, 0.20])
    vehicle_type = rng.choice(["sedan", "suv", "truck"], size=n_samples, p=[0.5, 0.3, 0.2])

    traffic_density = np.clip(rng.normal(0.55, 0.18, size=n_samples), 0.05, 0.95)
    avg_speed = np.clip(
        rng.normal(68, 9, size=n_samples)
        + np.where(road_type == "highway", 6, 0)
        - np.where(road_type == "urban", 5, 0)
        - night_ratio * 4,
        35,
        120,
    )
    experience_years = np.clip(rng.normal(8, 4, size=n_samples) - night_ratio * 2, 1, 35)
    previous_incidents = rng.poisson(0.2 + night_ratio * 0.35, size=n_samples)
    exposure_km = np.clip(35 + exposure_index * 12 + rng.normal(0, 6, size=n_samples), 10, 160)

    rapid_day = rng.poisson(0.6 + 0.9 * exposure_index * (1 - night_ratio * 0.7) + 0.4 * traffic_density)
    rapid_night = rng.poisson(0.2 + 0.8 * exposure_index * (0.3 + night_ratio) + 0.5 * traffic_density)
    sudden_day = rng.poisson(0.4 + 0.8 * exposure_index * (1 - night_ratio * 0.5) + 0.35 * traffic_density + 0.25 * (weather != "clear"))
    sudden_night = rng.poisson(0.2 + 0.9 * exposure_index * (0.25 + night_ratio) + 0.45 * traffic_density + 0.35 * (weather != "clear"))
    sharp_day = rng.poisson(0.3 + 0.6 * exposure_index * (1 - night_ratio * 0.6) + 0.25 * (road_type != "highway"))
    sharp_night = rng.poisson(0.15 + 0.55 * exposure_index * (0.2 + night_ratio) + 0.2 * (road_type != "highway"))
    overspeed_day = rng.poisson(0.7 + 1.1 * exposure_index * (1 - night_ratio * 0.5) + 0.35 * (avg_speed > 75))
    overspeed_night = rng.poisson(0.3 + 0.9 * exposure_index * (0.25 + night_ratio) + 0.3 * (avg_speed > 75))

    weather_rain = (weather == "rain").astype(float)
    weather_storm = (weather == "storm").astype(float)
    road_urban = (road_type == "urban").astype(float)
    road_suburban = (road_type == "suburban").astype(float)

    logit = (
        -3.25
        + 0.31 * rapid_day
        + 0.56 * rapid_night
        + 0.38 * sudden_day
        + 0.67 * sudden_night
        + 0.19 * sharp_day
        + 0.35 * sharp_night
        + 0.11 * overspeed_day
        + 0.18 * overspeed_night
        + 0.63 * night_ratio
        + 0.42 * traffic_density
        + 0.22 * ((avg_speed - 65) / 10)
        + 0.5 * weather_rain
        + 1.02 * weather_storm
        + 0.22 * road_urban
        + 0.13 * road_suburban
        + 0.32 * previous_incidents
        - 0.035 * experience_years
        + 0.18 * (exposure_km / 50 - 1)
    )

    prob = 1 / (1 + np.exp(-np.clip(logit, -15, 15)))
    prob = np.clip(prob, 0.01, 0.95)
    had_accident = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "rapid_acceleration_day": rapid_day,
            "rapid_acceleration_night": rapid_night,
            "sudden_stop_day": sudden_day,
            "sudden_stop_night": sudden_night,
            "sharp_turn_day": sharp_day,
            "sharp_turn_night": sharp_night,
            "over_speeding_day": overspeed_day,
            "over_speeding_night": overspeed_night,
            "night_ratio": night_ratio,
            "weather": weather,
            "road_type": road_type,
            "vehicle_type": vehicle_type,
            "traffic_density": traffic_density,
            "avg_speed": avg_speed,
            "experience_years": experience_years,
            "previous_incidents": previous_incidents,
            "exposure_km": exposure_km,
            "had_accident": had_accident,
        }
    )


def build_feature_frame(df: pd.DataFrame, events: Iterable[str]) -> pd.DataFrame:
    event_columns = [f"{event}{suffix}" for event in events for suffix in (DAY_SUFFIX, NIGHT_SUFFIX)]
    numeric_columns = [
        "night_ratio",
        "traffic_density",
        "avg_speed",
        "experience_years",
        "previous_incidents",
        "exposure_km",
    ]
    categorical_columns = ["weather", "road_type", "vehicle_type"]

    selected = df[event_columns + numeric_columns + categorical_columns]
    return pd.get_dummies(selected, drop_first=True, dtype=float)


def derive_event_weights(
    model: LogisticRegression,
    feature_names: List[str],
    events: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    coef = pd.Series(model.coef_[0], index=feature_names)

    day_raw = {event: abs(coef.get(f"{event}{DAY_SUFFIX}", 0.0)) for event in events}
    night_raw = {event: abs(coef.get(f"{event}{NIGHT_SUFFIX}", 0.0)) for event in events}

    scale_day = 2.5 / np.mean(list(day_raw.values())) if day_raw else 1.0
    scale_night = 3.6 / np.mean(list(night_raw.values())) if night_raw else 1.0

    day_weights = {
        event: round(value * scale_day, 2)
        for event, value in day_raw.items()
    }
    night_weights = {
        event: round(value * scale_night, 2)
        for event, value in night_raw.items()
    }

    return {
        "day": day_weights,
        "night": night_weights,
        "raw_coefficients": coef.to_dict(),
    }


def estimate_environment_multipliers(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    weather_baseline = df.loc[df["weather"] == "clear", "had_accident"].mean()
    weather_multiplier = {}
    for condition, group in df.groupby("weather"):
        rate = group["had_accident"].mean()
        weather_multiplier[condition] = float(rate / weather_baseline) if weather_baseline > 0 else 1.0

    road_baseline = df.loc[df["road_type"] == "highway", "had_accident"].mean()
    road_multiplier = {}
    for condition, group in df.groupby("road_type"):
        rate = group["had_accident"].mean()
        road_multiplier[condition] = float(rate / road_baseline) if road_baseline > 0 else 1.0

    night_bins = pd.cut(df["night_ratio"], bins=[-0.01, 0.2, 0.5, 1.0], labels=["low", "medium", "high"])
    night_rates = df.groupby(night_bins)["had_accident"].mean().to_dict()
    low_rate = night_rates.get("low", df["had_accident"].mean())
    night_multiplier = {
        key: float(value / low_rate) if low_rate > 0 else 1.0
        for key, value in night_rates.items()
    }
    for label in ["low", "medium", "high"]:
        night_multiplier.setdefault(label, 1.0)

    traffic = df["traffic_density"]
    low_q, high_q = traffic.quantile([0.25, 0.75])
    median = float(traffic.median())
    mid_mask = (traffic >= low_q) & (traffic <= high_q)
    base_rate = df.loc[mid_mask, "had_accident"].mean()
    high_rate = df.loc[traffic >= high_q, "had_accident"].mean()

    slope = 0.0
    if base_rate > 0 and high_rate > 0 and high_q > median:
        slope = (high_rate / base_rate - 1.0) / max(high_q - median, 1e-3)

    return {
        "weather": weather_multiplier,
        "road": road_multiplier,
        "night_bins": night_multiplier,
        "traffic": {
            "baseline": median,
            "slope": float(slope),
            "min_factor": 0.9,
            "max_factor": 1.45,
        },
        "cap": 1.65,
    }


def calculate_safety_scores(
    df: pd.DataFrame,
    event_weights: Dict[str, Dict[str, float]],
    env_multipliers: Dict[str, Dict[str, float]],
    events: Iterable[str],
) -> pd.DataFrame:
    day_weights = event_weights["day"]
    night_weights = event_weights["night"]
    weather_map = env_multipliers["weather"]
    road_map = env_multipliers["road"]
    night_map = env_multipliers["night_bins"]
    traffic_cfg = env_multipliers["traffic"]
    cap = env_multipliers.get("cap", 1.6)

    def env_factor(row: pd.Series) -> float:
        if row["night_ratio"] >= 0.5:
            night_bucket = "high"
        elif row["night_ratio"] >= 0.2:
            night_bucket = "medium"
        else:
            night_bucket = "low"

        weather_factor = weather_map.get(row["weather"], 1.0)
        road_factor = road_map.get(row["road_type"], 1.0)
        night_factor = night_map.get(night_bucket, 1.0)
        traffic_adjust = 1 + traffic_cfg["slope"] * (row["traffic_density"] - traffic_cfg["baseline"])
        traffic_factor = float(np.clip(traffic_adjust, traffic_cfg["min_factor"], traffic_cfg["max_factor"]))
        composite = weather_factor * road_factor * night_factor * traffic_factor
        return float(np.clip(composite, 1.0, cap))

    scored = df.copy()
    scored["environment_factor"] = scored.apply(env_factor, axis=1)

    penalty = np.zeros(len(scored))
    for event in events:
        penalty += scored[f"{event}{DAY_SUFFIX}"] * day_weights[event]
        penalty += scored[f"{event}{NIGHT_SUFFIX}"] * night_weights[event]

    scored["base_penalty"] = penalty
    scored["adjusted_penalty"] = penalty * scored["environment_factor"]
    scored["safety_score"] = np.clip(100 - scored["adjusted_penalty"], 0, 100)

    return scored


def _compute_youden(tp: int, fp: int, tn: int, fn: int) -> float:
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return tpr - fpr


def tune_grade_thresholds(df: pd.DataFrame) -> Dict[str, object]:
    scores = df["safety_score"].to_numpy()
    labels = df["had_accident"].to_numpy()

    aggressive_candidates = np.arange(50, 76, 0.5)
    safe_candidates = np.arange(70, 91, 0.5)

    best_aggressive, best_aggressive_j = 65.0, -1.0
    best_safe, best_safe_j = 80.0, -1.0

    for threshold in aggressive_candidates:
        pred = (scores <= threshold).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        j_score = _compute_youden(tp, fp, tn, fn)
        if j_score > best_aggressive_j:
            best_aggressive_j = j_score
            best_aggressive = float(threshold)

    inverse_labels = 1 - labels
    for threshold in safe_candidates:
        pred = (scores >= threshold).astype(int)
        tp = int(((pred == 1) & (inverse_labels == 1)).sum())
        fp = int(((pred == 1) & (inverse_labels == 0)).sum())
        fn = int(((pred == 0) & (inverse_labels == 1)).sum())
        tn = int(((pred == 0) & (inverse_labels == 0)).sum())
        j_score = _compute_youden(tp, fp, tn, fn)
        if j_score > best_safe_j:
            best_safe_j = j_score
            best_safe = float(threshold)

    if best_safe <= best_aggressive + 2:
        best_safe = min(best_aggressive + 5, 92)

    bins = [-np.inf, best_aggressive, best_safe, np.inf]
    labels_grade = ["AGGRESSIVE", "MODERATE", "SAFE"]
    df["grade"] = pd.cut(df["safety_score"], bins=bins, labels=labels_grade, include_lowest=True)

    grade_distribution = (
        df["grade"].value_counts(normalize=True).sort_index().round(4).to_dict()
    )
    grade_accident_rate = (
        df.groupby("grade")["had_accident"].mean().round(4).to_dict()
    )

    return {
        "aggressive_cut": round(best_aggressive, 1),
        "safe_cut": round(best_safe, 1),
        "moderate_range": [
            round(best_aggressive + 0.1, 1),
            round(best_safe - 0.1, 1),
        ],
        "grade_distribution": grade_distribution,
        "grade_accident_rate": grade_accident_rate,
    }


def _calc_metrics(name: str, y_true, y_pred, y_prob) -> ModelMetrics:
    return ModelMetrics(
        name=name,
        accuracy=round(accuracy_score(y_true, y_pred), 4),
        precision=round(precision_score(y_true, y_pred, zero_division=0), 4),
        recall=round(recall_score(y_true, y_pred, zero_division=0), 4),
        f1=round(f1_score(y_true, y_pred, zero_division=0), 4),
        auc=round(roc_auc_score(y_true, y_prob), 4),
    )


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    logistic_model: LogisticRegression,
) -> Dict[str, Dict[str, float]]:
    metrics = {}

    lr_prob = logistic_model.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    metrics["logistic_regression"] = _calc_metrics("logistic_regression", y_test, lr_pred, lr_prob).to_dict()

    xgb_model = XGBClassifier(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        reg_lambda=1.0,
    )
    xgb_model.fit(X_train, y_train)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)
    metrics["xgboost"] = _calc_metrics("xgboost", y_test, xgb_pred, xgb_prob).to_dict()

    lgb_model = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgb_model.fit(X_train, y_train)
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = (lgb_prob >= 0.5).astype(int)
    metrics["lightgbm"] = _calc_metrics("lightgbm", y_test, lgb_pred, lgb_prob).to_dict()

    return metrics


def run_scenario(
    df: pd.DataFrame,
    env_multipliers: Dict[str, Dict[str, float]],
    events: Tuple[str, ...],
) -> Dict[str, object]:
    feature_frame = build_feature_frame(df, events)
    target = df["had_accident"]

    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=0.25,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    logistic_model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    logistic_model.fit(X_train, y_train)

    model_metrics = evaluate_models(X_train, X_test, y_train, y_test, logistic_model)

    logistic_full = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    logistic_full.fit(feature_frame, target)

    event_weights = derive_event_weights(logistic_full, list(feature_frame.columns), events)
    scored_df = calculate_safety_scores(df, event_weights, env_multipliers, events)
    thresholds = tune_grade_thresholds(scored_df)

    score_stats = (
        scored_df["safety_score"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict()
    )

    return {
        "event_weights": {
            "day": event_weights["day"],
            "night": event_weights["night"],
        },
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
        "model_metrics": model_metrics,
        "mean_events_per_record": {
            event: round(float((df[f"{event}{DAY_SUFFIX}"] + df[f"{event}{NIGHT_SUFFIX}"]).mean()), 3)
            for event in events
        },
    }


def build_comparison(scenarios: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    comparison = {}
    a = scenarios["with_overspeed"]
    b = scenarios["without_overspeed"]

    def diff(metric: str, model: str) -> float:
        return round(
            b["model_metrics"][model][metric] - a["model_metrics"][model][metric],
            4,
        )

    comparison["model_deltas"] = {
        model: {
            metric: diff(metric, model)
            for metric in ("auc", "f1", "recall", "precision", "accuracy")
        }
        for model in ("logistic_regression", "xgboost", "lightgbm")
    }

    def grade_delta(grade: str) -> float:
        return round(
            b["score_summary"]["grade_distribution"].get(grade, 0.0)
            - a["score_summary"]["grade_distribution"].get(grade, 0.0),
            4,
        )

    comparison["grade_distribution_delta"] = {
        grade: grade_delta(grade)
        for grade in ("SAFE", "MODERATE", "AGGRESSIVE")
    }

    comparison["threshold_delta"] = {
        key: round(b["thresholds"][key] - a["thresholds"][key], 2)
        if not isinstance(b["thresholds"][key], list)
        else [
            round(b["thresholds"][key][0] - a["thresholds"][key][0], 2),
            round(b["thresholds"][key][1] - a["thresholds"][key][1], 2),
        ]
        for key in ("aggressive_cut", "safe_cut", "moderate_range")
    }

    return comparison


def main(output_path: Path = Path("research/phase2_results.json")) -> Dict[str, object]:
    df = generate_phase2_dataset()

    dataset_summary = {
        "records": int(len(df)),
        "accident_rate": round(float(df["had_accident"].mean()), 4),
        "mean_night_ratio": round(float(df["night_ratio"].mean()), 4),
    }

    env_multipliers = estimate_environment_multipliers(df)

    scenario_outputs = {}
    for key, config in SCENARIOS.items():
        scenario_outputs[key] = {
            "label": config["label"],
            **run_scenario(df, env_multipliers, config["events"]),
        }

    comparison = build_comparison(scenario_outputs)

    results = {
        "dataset": dataset_summary,
        "environment_factors": env_multipliers,
        "scenarios": scenario_outputs,
        "comparison": comparison,
    }

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("Phase 2 scenarios completed:")
    print(f"  Records: {dataset_summary['records']} | Accident rate: {dataset_summary['accident_rate']:.2%}")

    for key, scenario in scenario_outputs.items():
        print(f"\n{scenario['label']}")
        weights = scenario["event_weights"]
        for event, day_weight in weights["day"].items():
            night_weight = weights["night"][event]
            print(f"  {event}: day -{day_weight}, night -{night_weight}")
        thresholds = scenario["thresholds"]
        print(
            f"  Thresholds => Aggressive <= {thresholds['aggressive_cut']} | "
            f"Moderate ({thresholds['moderate_range'][0]}-{thresholds['moderate_range'][1]}) | "
            f"Safe >= {thresholds['safe_cut']}"
        )
        print("  Model AUC (LR / XGB / LGBM):",
              scenario["model_metrics"]["logistic_regression"]["auc"],
              scenario["model_metrics"]["xgboost"]["auc"],
              scenario["model_metrics"]["lightgbm"]["auc"],
        )

    return results


if __name__ == "__main__":
    main()
