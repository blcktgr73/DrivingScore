#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-E: 주간/야간 Weight 분석
=================================

주간과 야간을 구분하여 각각 모델을 학습하고
이벤트별 가중치를 비교 분석합니다.

감점 시스템에 적용할 가중치를 도출합니다.

작성일: 2025-10-15
"""

import os
import sys
import json
import random
import math
from datetime import datetime

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 100)
print(" Phase 4-E: 주간/야간 Weight 분석")
print("=" * 100)
print()

# ============================================================================
# Logistic Regression with Class Weight
# ============================================================================

class LogisticRegressionWithClassWeight:
    """Class Weight 적용 Logistic Regression"""
    def __init__(self, learning_rate=0.01, iterations=1000, class_weight='balanced'):
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
            self.weight_positive = n_samples / (2 * n_positive) if n_positive > 0 else 1.0
            self.weight_negative = n_samples / (2 * n_negative) if n_negative > 0 else 1.0

        for iteration in range(self.iterations):
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
        predictions = []
        for i in range(len(X)):
            z = sum(X[i][j] * self.weights[j] for j in range(len(self.weights))) + self.bias
            predictions.append(self.sigmoid(z))
        return predictions

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probas]

# ============================================================================
# 평가 함수
# ============================================================================

def evaluate_model(y_true, y_pred, y_proba):
    """모델 평가"""
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    start_time = datetime.now()
    print(f"⏰ 분석 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 데이터 로드
    print("=" * 100)
    print("Step 1: 데이터 로드")
    print("=" * 100)
    print()

    with open("research/phase4e_combined_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    combined_data = data['data']
    print(f"  ✅ 로드 완료: {len(combined_data):,}개\n")

    # 2. 주간/야간 데이터 분리
    print("=" * 100)
    print("Step 2: 주간/야간 데이터 분리")
    print("=" * 100)
    print()

    day_data = [d for d in combined_data if d['features']['is_night'] == 0]
    night_data = [d for d in combined_data if d['features']['is_night'] == 1]

    print(f"  주간 데이터: {len(day_data):,}개 ({len(day_data)/len(combined_data)*100:.1f}%)")
    print(f"  야간 데이터: {len(night_data):,}개 ({len(night_data)/len(combined_data)*100:.1f}%)")

    # 3. 주간 모델 학습 (Scenario A)
    print("\n" + "=" * 100)
    print("Step 3: 주간 모델 학습 (Scenario A)")
    print("=" * 100)
    print()

    # Train/Test 분할
    random.seed(42)
    random.shuffle(day_data)
    split_idx = int(len(day_data) * 0.75)
    day_train = day_data[:split_idx]
    day_test = day_data[split_idx:]

    print(f"  Train: {len(day_train):,}개")
    print(f"  Test:  {len(day_test):,}개")

    X_train_day_a = [[d['features']['rapid_accel'], d['features']['sudden_stop'],
                      d['features']['sharp_turn'], d['features']['over_speed']]
                     for d in day_train]
    y_train_day = [d['label'] for d in day_train]

    X_test_day_a = [[d['features']['rapid_accel'], d['features']['sudden_stop'],
                     d['features']['sharp_turn'], d['features']['over_speed']]
                    for d in day_test]
    y_test_day = [d['label'] for d in day_test]

    model_day_a = LogisticRegressionWithClassWeight(learning_rate=0.01, iterations=500, class_weight='balanced')
    model_day_a.fit(X_train_day_a, y_train_day)

    y_pred_day_a = model_day_a.predict(X_test_day_a)
    y_proba_day_a = model_day_a.predict_proba(X_test_day_a)
    metrics_day_a = evaluate_model(y_test_day, y_pred_day_a, y_proba_day_a)

    print(f"\n  학습된 가중치:")
    print(f"    급가속: {model_day_a.weights[0]:>8.3f}")
    print(f"    급정거: {model_day_a.weights[1]:>8.3f}")
    print(f"    급회전: {model_day_a.weights[2]:>8.3f}")
    print(f"    과속:   {model_day_a.weights[3]:>8.3f}")
    print(f"    Bias:   {model_day_a.bias:>8.3f}")

    print(f"\n  성능:")
    print(f"    F1 Score:  {metrics_day_a['f1']:.3f}")
    print(f"    Recall:    {metrics_day_a['recall']:.3f}")
    print(f"    Precision: {metrics_day_a['precision']:.3f}")

    # 4. 야간 모델 학습 (Scenario A)
    print("\n" + "=" * 100)
    print("Step 4: 야간 모델 학습 (Scenario A)")
    print("=" * 100)
    print()

    random.seed(42)
    random.shuffle(night_data)
    split_idx = int(len(night_data) * 0.75)
    night_train = night_data[:split_idx]
    night_test = night_data[split_idx:]

    print(f"  Train: {len(night_train):,}개")
    print(f"  Test:  {len(night_test):,}개")

    X_train_night_a = [[d['features']['rapid_accel'], d['features']['sudden_stop'],
                        d['features']['sharp_turn'], d['features']['over_speed']]
                       for d in night_train]
    y_train_night = [d['label'] for d in night_train]

    X_test_night_a = [[d['features']['rapid_accel'], d['features']['sudden_stop'],
                       d['features']['sharp_turn'], d['features']['over_speed']]
                      for d in night_test]
    y_test_night = [d['label'] for d in night_test]

    model_night_a = LogisticRegressionWithClassWeight(learning_rate=0.01, iterations=500, class_weight='balanced')
    model_night_a.fit(X_train_night_a, y_train_night)

    y_pred_night_a = model_night_a.predict(X_test_night_a)
    y_proba_night_a = model_night_a.predict_proba(X_test_night_a)
    metrics_night_a = evaluate_model(y_test_night, y_pred_night_a, y_proba_night_a)

    print(f"\n  학습된 가중치:")
    print(f"    급가속: {model_night_a.weights[0]:>8.3f}")
    print(f"    급정거: {model_night_a.weights[1]:>8.3f}")
    print(f"    급회전: {model_night_a.weights[2]:>8.3f}")
    print(f"    과속:   {model_night_a.weights[3]:>8.3f}")
    print(f"    Bias:   {model_night_a.bias:>8.3f}")

    print(f"\n  성능:")
    print(f"    F1 Score:  {metrics_night_a['f1']:.3f}")
    print(f"    Recall:    {metrics_night_a['recall']:.3f}")
    print(f"    Precision: {metrics_night_a['precision']:.3f}")

    # 5. 주간 vs 야간 비교
    print("\n" + "=" * 100)
    print("Step 5: 주간 vs 야간 Weight 비교")
    print("=" * 100)
    print()

    print(f"{'이벤트':>8s}  {'주간 Weight':>12s}  {'야간 Weight':>12s}  {'차이':>12s}  {'비율 (야간/주간)':>18s}")
    print("-" * 85)

    events = ['급가속', '급정거', '급회전', '과속']
    for i, event in enumerate(events):
        day_w = model_day_a.weights[i]
        night_w = model_night_a.weights[i]
        diff = night_w - day_w
        ratio = abs(night_w) / abs(day_w) if abs(day_w) > 0.001 else 0

        print(f"{event:>8s}  {day_w:>12.3f}  {night_w:>12.3f}  {diff:>+12.3f}  {ratio:>18.2f}x")

    # 6. 감점 시스템 권장 가중치
    print("\n" + "=" * 100)
    print("Step 6: 감점 시스템 권장 가중치")
    print("=" * 100)
    print()

    # 절대값 기준으로 정규화
    day_weights_abs = [abs(w) for w in model_day_a.weights]
    night_weights_abs = [abs(w) for w in model_night_a.weights]

    max_day = max(day_weights_abs) if day_weights_abs else 1
    max_night = max(night_weights_abs) if night_weights_abs else 1

    day_normalized = [w / max_day * 5 for w in day_weights_abs]  # 1~5점 스케일
    night_normalized = [w / max_night * 5 for w in night_weights_abs]

    print(f"  주간 감점 (1~5점 스케일):")
    for i, event in enumerate(events):
        penalty = round(day_normalized[i])
        print(f"    {event}: {penalty}점")

    print(f"\n  야간 감점 (1~5점 스케일):")
    for i, event in enumerate(events):
        penalty = round(night_normalized[i])
        print(f"    {event}: {penalty}점")

    print(f"\n  야간 가중치 배율 (주간 대비):")
    for i, event in enumerate(events):
        if day_normalized[i] > 0:
            multiplier = night_normalized[i] / day_normalized[i]
            print(f"    {event}: {multiplier:.2f}x")

    # 7. Scenario B 분석
    print("\n" + "=" * 100)
    print("Step 7: Scenario B 분석 (3개 이벤트)")
    print("=" * 100)
    print()

    # 주간 Scenario B
    X_train_day_b = [[d['features']['rapid_accel'], d['features']['sudden_stop'],
                      d['features']['sharp_turn']] for d in day_train]
    X_test_day_b = [[d['features']['rapid_accel'], d['features']['sudden_stop'],
                     d['features']['sharp_turn']] for d in day_test]

    model_day_b = LogisticRegressionWithClassWeight(learning_rate=0.01, iterations=500, class_weight='balanced')
    model_day_b.fit(X_train_day_b, y_train_day)

    y_pred_day_b = model_day_b.predict(X_test_day_b)
    y_proba_day_b = model_day_b.predict_proba(X_test_day_b)
    metrics_day_b = evaluate_model(y_test_day, y_pred_day_b, y_proba_day_b)

    print(f"  주간 (Scenario B):")
    print(f"    급가속: {model_day_b.weights[0]:>8.3f}")
    print(f"    급정거: {model_day_b.weights[1]:>8.3f}")
    print(f"    급회전: {model_day_b.weights[2]:>8.3f}")
    print(f"    F1: {metrics_day_b['f1']:.3f}")

    # 야간 Scenario B
    X_train_night_b = [[d['features']['rapid_accel'], d['features']['sudden_stop'],
                        d['features']['sharp_turn']] for d in night_train]
    X_test_night_b = [[d['features']['rapid_accel'], d['features']['sudden_stop'],
                       d['features']['sharp_turn']] for d in night_test]

    model_night_b = LogisticRegressionWithClassWeight(learning_rate=0.01, iterations=500, class_weight='balanced')
    model_night_b.fit(X_train_night_b, y_train_night)

    y_pred_night_b = model_night_b.predict(X_test_night_b)
    y_proba_night_b = model_night_b.predict_proba(X_test_night_b)
    metrics_night_b = evaluate_model(y_test_night, y_pred_night_b, y_proba_night_b)

    print(f"\n  야간 (Scenario B):")
    print(f"    급가속: {model_night_b.weights[0]:>8.3f}")
    print(f"    급정거: {model_night_b.weights[1]:>8.3f}")
    print(f"    급회전: {model_night_b.weights[2]:>8.3f}")
    print(f"    F1: {metrics_night_b['f1']:.3f}")

    # 8. 결과 저장
    results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "phase": "4E",
            "day_samples": len(day_data),
            "night_samples": len(night_data)
        },
        "scenario_a": {
            "day": {
                "weights": {
                    "rapid_accel": model_day_a.weights[0],
                    "sudden_stop": model_day_a.weights[1],
                    "sharp_turn": model_day_a.weights[2],
                    "over_speed": model_day_a.weights[3],
                    "bias": model_day_a.bias
                },
                "metrics": metrics_day_a
            },
            "night": {
                "weights": {
                    "rapid_accel": model_night_a.weights[0],
                    "sudden_stop": model_night_a.weights[1],
                    "sharp_turn": model_night_a.weights[2],
                    "over_speed": model_night_a.weights[3],
                    "bias": model_night_a.bias
                },
                "metrics": metrics_night_a
            }
        },
        "scenario_b": {
            "day": {
                "weights": {
                    "rapid_accel": model_day_b.weights[0],
                    "sudden_stop": model_day_b.weights[1],
                    "sharp_turn": model_day_b.weights[2],
                    "bias": model_day_b.bias
                },
                "metrics": metrics_day_b
            },
            "night": {
                "weights": {
                    "rapid_accel": model_night_b.weights[0],
                    "sudden_stop": model_night_b.weights[1],
                    "sharp_turn": model_night_b.weights[2],
                    "bias": model_night_b.bias
                },
                "metrics": metrics_night_b
            }
        },
        "penalty_recommendation": {
            "day": {
                "rapid_accel": round(day_normalized[0]),
                "sudden_stop": round(day_normalized[1]),
                "sharp_turn": round(day_normalized[2]),
                "over_speed": round(day_normalized[3])
            },
            "night": {
                "rapid_accel": round(night_normalized[0]),
                "sudden_stop": round(night_normalized[1]),
                "sharp_turn": round(night_normalized[2]),
                "over_speed": round(night_normalized[3])
            }
        }
    }

    output_file = "research/phase4e_day_night_weights.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ 결과 파일 저장: {output_file}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n⏱️  총 소요 시간: {duration:.1f}초")
    print()

if __name__ == "__main__":
    main()
