#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 Rebaseline (A/B) under new detection criteria.

Generates synthetic short windows per sample and derives event counts using:
- Rapid Accel/Decel: Δspeed ≥ 10 km/h/s sustained for 3 seconds
- Sharp Turn: Centrifugal Acceleration Jump ≥ 400 degree m/s^2

Then trains a simple logistic regression with class weights (as in Phase 4-D)
and evaluates both Scenario A (4 events) and Scenario B (3 events).
"""

import os
import sys
import json
import math
import random
from typing import List, Tuple, Dict

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(SCRIPT_DIR)
from detection_criteria import (
    count_rapid_accel_events_kmh,
    count_rapid_decel_events_kmh,
    count_sharp_turn_events_jump,
)

def normal_random(mean_val: float, std_val: float) -> float:
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mean_val + z * std_val

class LogisticRegressionWithClassWeight:
    def __init__(self, learning_rate=0.01, iterations=600, class_weight='balanced'):
        self.lr = learning_rate
        self.iterations = iterations
        self.class_weight = class_weight
        self.weights = None
        self.bias = None
        self.weight_positive = 1.0
        self.weight_negative = 1.0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-min(max(z, -500), 500)))

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        if self.class_weight == 'balanced':
            n_positive = sum(y)
            n_negative = n_samples - n_positive
            self.weight_positive = n_samples / (2 * n_positive)
            self.weight_negative = n_samples / (2 * n_negative)

        for _ in range(self.iterations):
            predictions = []
            for i in range(n_samples):
                z = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                predictions.append(self.sigmoid(z))

            dw = [0.0] * n_features
            db = 0.0
            for i in range(n_samples):
                error = predictions[i] - y[i]
                weight = self.weight_positive if y[i] == 1 else self.weight_negative
                for j in range(n_features):
                    dw[j] += weight * error * X[i][j]
                db += weight * error

            for j in range(n_features):
                self.weights[j] -= self.lr * dw[j] / n_samples
            self.bias -= self.lr * db / n_samples

    def predict_proba(self, X):
        out = []
        for i in range(len(X)):
            z = sum(X[i][j] * self.weights[j] for j in range(len(self.weights))) + self.bias
            out.append(self.sigmoid(z))
        return out

def metrics_at_threshold(y_true, y_proba, threshold: float) -> Dict[str, float]:
    y_pred = [1 if p >= threshold else 0 for p in y_proba]
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}

def find_best_threshold(y_true, y_proba, mode: str) -> Tuple[float, Dict[str, float]]:
    best_t, best_m = 0.5, {"precision":0, "recall":0, "f1":0}
    for t in [i/100 for i in range(30, 90)]:
        m = metrics_at_threshold(y_true, y_proba, t)
        score = m[mode]
        if score > best_m.get(mode, 0):
            best_m = m
            best_t = t
    return best_t, best_m

def generate_dataset(n_samples=15000, accident_rate=0.359) -> Tuple[List[List[int]], List[int]]:
    X, y = [], []
    random.seed(42)
    for _ in range(n_samples):
        is_accident = 1 if random.random() < accident_rate else 0

        # Baseline profile means/stdev differ by label to create separation
        if is_accident:
            accx_mu, accx_std = 1.6, 0.8
            gyroz_mu, gyroz_std = 1.2, 0.6
            speed_mu, speed_std = 72.0, 15.0
        else:
            accx_mu, accx_std = 0.6, 0.5
            gyroz_mu, gyroz_std = 0.4, 0.3
            speed_mu, speed_std = 54.0, 12.0

        accx = normal_random(accx_mu, accx_std)
        gyroz = normal_random(gyroz_mu, gyroz_std)
        speed = max(0.0, normal_random(speed_mu, speed_std))

        # build short 4s window @1Hz
        sampling_hz = 1.0
        speeds_kmh = []
        s = speed
        for _ in range(4):
            s = max(0.0, s + accx * 3.6 + random.uniform(-0.8, 0.8))
            speeds_kmh.append(s)
        accy = normal_random(0, 0.3)
        accz = normal_random(9.8, 0.2)
        accel_mag = math.sqrt(accx**2 + accy**2 + accz**2)
        gyro_series = [gyroz + random.uniform(-0.2, 0.2) for _ in range(4)]
        accel_series = [accel_mag + random.uniform(-0.2, 0.2) for _ in range(4)]

        ra = count_rapid_accel_events_kmh(speeds_kmh, sampling_hz)
        sd = count_rapid_decel_events_kmh(speeds_kmh, sampling_hz)
        st = count_sharp_turn_events_jump(gyro_series, accel_series, sampling_hz)

        # simple overspeed proxy: avg speed > 100km/h
        os_ev = 1 if (sum(speeds_kmh)/len(speeds_kmh)) > 100 else 0

        X.append([ra, sd, st, os_ev])
        y.append(is_accident)
    return X, y

def run_scenario(events: List[int]) -> Dict[str, Dict[str, float]]:
    X, y = generate_dataset()
    # Prepare X for scenario
    if len(events) == 3:
        X = [[row[0], row[1], row[2]] for row in X]
    model = LogisticRegressionWithClassWeight()
    model.fit(X, y)
    proba = model.predict_proba(X)
    # Strategies
    t_f1, m_f1 = find_best_threshold(y, proba, 'f1')
    t_p60, m_p60 = find_best_threshold(y, proba, 'precision')
    return {
        'best_f1': {'threshold': t_f1, 'metrics': m_f1},
        'best_precision': {'threshold': t_p60, 'metrics': m_p60},
    }

def main():
    print("="*80)
    print(" Phase 4 Rebaseline under new criteria (A/B)")
    print("="*80)
    results = {}
    # Scenario A: 4 events
    results['scenario_a'] = run_scenario([0,1,2,3])
    # Scenario B: 3 events
    results['scenario_b'] = run_scenario([0,1,2])
    out = os.path.join(SCRIPT_DIR, 'phase4_rebaseline_new_criteria.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out}")

if __name__ == '__main__':
    main()

