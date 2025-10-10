#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-D Step 1: Class Weight 적용
====================================

Phase 4-C의 Logistic Regression에 Class Weight를 적용하여
Class Imbalance 문제를 해결합니다.

목표:
- Recall: 6.2% → 25%
- Precision: 73.9% → 60%
- F1: 11.4% → 35%

작성일: 2025-10-10
"""

import os
import sys
import json
import random
import math
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print(" Phase 4-D Step 1: Class Weight 적용")
print("=" * 80)
print()

# ============================================================================
# 유틸리티 함수 (Phase 4-C에서 가져옴)
# ============================================================================

def mean(data):
    return sum(data) / len(data) if data else 0

def std(data):
    if not data:
        return 0
    m = mean(data)
    variance = sum((x - m) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

def normal_random(mean_val, std_val):
    """박스-뮬러 변환"""
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mean_val + z * std_val

def correlation(x, y):
    """피어슨 상관계수"""
    if len(x) != len(y) or len(x) == 0:
        return 0
    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = math.sqrt(sum((x[i] - mean_x)**2 for i in range(n)) *
                           sum((y[i] - mean_y)**2 for i in range(n)))
    return numerator / denominator if denominator != 0 else 0

def calculate_distance_km(lat1, lon1, lat2, lon2):
    """거리 계산 (km)"""
    R = 6371
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ============================================================================
# 개선된 Logistic Regression (Class Weight 적용)
# ============================================================================

class LogisticRegressionWithClassWeight:
    """Class Weight를 적용한 로지스틱 회귀"""
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

        # Class weight 계산
        if self.class_weight == 'balanced':
            n_positive = sum(y)
            n_negative = n_samples - n_positive

            # Balanced weight: n_samples / (n_classes * n_samples_for_class)
            self.weight_positive = n_samples / (2 * n_positive)
            self.weight_negative = n_samples / (2 * n_negative)

            print(f"\n[Class Weight 적용]")
            print(f"  사고 샘플: {n_positive}개 (가중치: {self.weight_positive:.4f})")
            print(f"  비사고 샘플: {n_negative}개 (가중치: {self.weight_negative:.4f})")
            print(f"  → 사고 샘플에 {self.weight_positive/self.weight_negative:.2f}배 가중치 부여")

        for iteration in range(self.iterations):
            # Forward pass
            predictions = []
            for i in range(n_samples):
                z = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                predictions.append(self.sigmoid(z))

            # Compute gradients with class weights
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n_samples):
                error = predictions[i] - y[i]

                # Class weight 적용
                weight = self.weight_positive if y[i] == 1 else self.weight_negative

                for j in range(n_features):
                    dw[j] += weight * error * X[i][j]
                db += weight * error

            # Update weights
            for j in range(n_features):
                self.weights[j] -= self.lr * dw[j] / n_samples
            self.bias -= self.lr * db / n_samples

            # Progress
            if (iteration + 1) % 200 == 0:
                print(f"  Iteration {iteration + 1}/{self.iterations} 완료")

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
# 기존 Logistic Regression (Phase 4-C)
# ============================================================================

class LogisticRegression:
    """기존 로지스틱 회귀 (비교용)"""
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-min(max(z, -500), 500)))

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for iteration in range(self.iterations):
            predictions = []
            for i in range(n_samples):
                z = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                predictions.append(self.sigmoid(z))

            dw = [0.0] * n_features
            db = 0.0
            for i in range(n_samples):
                error = predictions[i] - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error

            for j in range(n_features):
                self.weights[j] -= self.lr * dw[j] / n_samples
            self.bias -= self.lr * db / n_samples

            if (iteration + 1) % 200 == 0:
                print(f"  Iteration {iteration + 1}/{self.iterations} 완료")

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
# 평가 메트릭
# ============================================================================

def calculate_metrics(y_true, y_pred, y_proba):
    """평가 지표 계산"""
    # Confusion matrix
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

    # Metrics
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # AUC (간단한 사다리꼴 근사)
    sorted_pairs = sorted(zip(y_proba, y_true), reverse=True)
    positives = sum(y_true)
    negatives = len(y_true) - positives

    if positives == 0 or negatives == 0:
        auc = 0.5
    else:
        true_positive = 0
        false_positive = 0
        auc_sum = 0.0
        prev_fpr = 0.0

        for prob, label in sorted_pairs:
            if label == 1:
                true_positive += 1
            else:
                false_positive += 1
                tpr = true_positive / positives
                fpr = false_positive / negatives
                auc_sum += (fpr - prev_fpr) * tpr
                prev_fpr = fpr

        auc = auc_sum

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }

# ============================================================================
# 데이터 생성 (Phase 4-C와 동일)
# ============================================================================

def generate_data(n_samples=15000, accident_rate=0.359):
    """
    Phase 4-C와 동일한 데이터 생성
    (재현성을 위해 간소화된 버전)
    """
    print("\n" + "=" * 80)
    print("📊 데이터 생성 (Phase 4-C 기준)")
    print("=" * 80)

    random.seed(42)  # 재현성

    X = []
    y = []

    for i in range(n_samples):
        # 사고 여부 결정
        is_accident = 1 if random.random() < accident_rate else 0

        # 사고인 경우 이벤트가 더 많음
        if is_accident:
            rapid_accel = max(0, int(normal_random(5.0, 2.0)))
            sudden_stop = max(0, int(normal_random(6.0, 2.5)))
            sharp_turn = max(0, int(normal_random(4.0, 2.0)))
            over_speed = max(0, int(normal_random(4.5, 2.0)))
        else:
            rapid_accel = max(0, int(normal_random(2.0, 1.5)))
            sudden_stop = max(0, int(normal_random(2.5, 1.5)))
            sharp_turn = max(0, int(normal_random(1.5, 1.0)))
            over_speed = max(0, int(normal_random(2.0, 1.5)))

        X.append([rapid_accel, sudden_stop, sharp_turn, over_speed])
        y.append(is_accident)

    print(f"\n총 샘플: {n_samples:,}개")
    print(f"사고: {sum(y):,}개 ({sum(y)/len(y)*100:.1f}%)")
    print(f"비사고: {len(y)-sum(y):,}개 ({(len(y)-sum(y))/len(y)*100:.1f}%)")

    return X, y

def train_test_split(X, y, test_size=0.25):
    """Train/Test 분할"""
    random.seed(42)
    indices = list(range(len(X)))
    random.shuffle(indices)

    split_idx = int(len(X) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    # 1. 데이터 생성
    X, y = generate_data(n_samples=15000, accident_rate=0.359)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    print(f"\nTrain: {len(X_train):,}개, Test: {len(X_test):,}개")

    results = {}

    # 2. Phase 4-C 모델 (기존 - 비교용)
    print("\n" + "=" * 80)
    print("🔄 Phase 4-C: 기존 Logistic Regression (Class Weight 없음)")
    print("=" * 80)

    model_baseline = LogisticRegression(learning_rate=0.01, iterations=500)
    model_baseline.fit(X_train, y_train)

    y_proba_baseline = model_baseline.predict_proba(X_test)
    y_pred_baseline = model_baseline.predict(X_test, threshold=0.5)

    metrics_baseline = calculate_metrics(y_test, y_pred_baseline, y_proba_baseline)

    print("\n[Phase 4-C 결과]")
    print(f"  Accuracy:  {metrics_baseline['accuracy']:.4f}")
    print(f"  Precision: {metrics_baseline['precision']:.4f}")
    print(f"  Recall:    {metrics_baseline['recall']:.4f}")
    print(f"  F1 Score:  {metrics_baseline['f1']:.4f}")
    print(f"  AUC:       {metrics_baseline['auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = metrics_baseline['confusion_matrix']
    print(f"    TP: {cm['tp']}, FP: {cm['fp']}")
    print(f"    FN: {cm['fn']}, TN: {cm['tn']}")

    results['phase4c_baseline'] = metrics_baseline

    # 3. Phase 4-D Step 1: Class Weight 적용
    print("\n" + "=" * 80)
    print("✨ Phase 4-D Step 1: Class Weight 적용")
    print("=" * 80)

    model_weighted = LogisticRegressionWithClassWeight(
        learning_rate=0.01,
        iterations=500,
        class_weight='balanced'
    )
    model_weighted.fit(X_train, y_train)

    y_proba_weighted = model_weighted.predict_proba(X_test)
    y_pred_weighted = model_weighted.predict(X_test, threshold=0.5)

    metrics_weighted = calculate_metrics(y_test, y_pred_weighted, y_proba_weighted)

    print("\n[Phase 4-D Step 1 결과]")
    print(f"  Accuracy:  {metrics_weighted['accuracy']:.4f}")
    print(f"  Precision: {metrics_weighted['precision']:.4f} (4-C: {metrics_baseline['precision']:.4f})")
    print(f"  Recall:    {metrics_weighted['recall']:.4f} (4-C: {metrics_baseline['recall']:.4f})")
    print(f"  F1 Score:  {metrics_weighted['f1']:.4f} (4-C: {metrics_baseline['f1']:.4f})")
    print(f"  AUC:       {metrics_weighted['auc']:.4f} (4-C: {metrics_baseline['auc']:.4f})")
    print(f"\n  Confusion Matrix:")
    cm_w = metrics_weighted['confusion_matrix']
    print(f"    TP: {cm_w['tp']}, FP: {cm_w['fp']}")
    print(f"    FN: {cm_w['fn']}, TN: {cm_w['tn']}")

    results['phase4d_step1_class_weight'] = metrics_weighted

    # 4. 개선 분석
    print("\n" + "=" * 80)
    print("📈 개선 분석")
    print("=" * 80)

    recall_improvement = (metrics_weighted['recall'] - metrics_baseline['recall']) / metrics_baseline['recall'] * 100
    f1_improvement = (metrics_weighted['f1'] - metrics_baseline['f1']) / metrics_baseline['f1'] * 100
    precision_change = (metrics_weighted['precision'] - metrics_baseline['precision']) / metrics_baseline['precision'] * 100

    print(f"\nRecall 개선:")
    print(f"  {metrics_baseline['recall']:.1%} → {metrics_weighted['recall']:.1%}")
    print(f"  +{recall_improvement:.1f}% 향상")

    print(f"\nF1 Score 개선:")
    print(f"  {metrics_baseline['f1']:.4f} → {metrics_weighted['f1']:.4f}")
    print(f"  +{f1_improvement:.1f}% 향상")

    print(f"\nPrecision 변화 (Trade-off):")
    print(f"  {metrics_baseline['precision']:.1%} → {metrics_weighted['precision']:.1%}")
    if precision_change < 0:
        print(f"  {precision_change:.1f}% 하락 (예상된 Trade-off)")
    else:
        print(f"  +{precision_change:.1f}% 향상")

    print(f"\nAUC 변화:")
    print(f"  {metrics_baseline['auc']:.4f} → {metrics_weighted['auc']:.4f}")
    print(f"  {'+' if metrics_weighted['auc'] >= metrics_baseline['auc'] else ''}{(metrics_weighted['auc'] - metrics_baseline['auc']):.4f}")

    # 5. 결과 저장
    results['improvements'] = {
        'recall_improvement_pct': recall_improvement,
        'f1_improvement_pct': f1_improvement,
        'precision_change_pct': precision_change,
        'auc_change': metrics_weighted['auc'] - metrics_baseline['auc']
    }

    results['summary'] = {
        'goal_recall': 0.25,
        'achieved_recall': metrics_weighted['recall'],
        'goal_f1': 0.35,
        'achieved_f1': metrics_weighted['f1'],
        'goal_precision': 0.60,
        'achieved_precision': metrics_weighted['precision']
    }

    output_file = os.path.join(os.path.dirname(__file__), 'phase4d_step1_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {output_file}")

    # 6. 목표 달성 여부
    print("\n" + "=" * 80)
    print("🎯 목표 달성 여부")
    print("=" * 80)

    goals = [
        ("Recall ≥ 25%", metrics_weighted['recall'] >= 0.25, f"{metrics_weighted['recall']:.1%}"),
        ("F1 Score ≥ 0.35", metrics_weighted['f1'] >= 0.35, f"{metrics_weighted['f1']:.4f}"),
        ("Precision ≥ 60%", metrics_weighted['precision'] >= 0.60, f"{metrics_weighted['precision']:.1%}"),
    ]

    for goal, achieved, value in goals:
        status = "✅" if achieved else "❌"
        print(f"  {status} {goal}: {value}")

    print("\n" + "=" * 80)
    print("✅ Phase 4-D Step 1 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()
