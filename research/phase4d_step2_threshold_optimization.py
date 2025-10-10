#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-D Step 2: Threshold 최적화
===================================

Class Weight를 적용한 모델에서 최적 Threshold를 탐색하여
Precision-Recall 균형을 맞춥니다.

목표:
- F1 Score 최대화
- Precision ≥ 60%, Recall ≥ 25%
- 목표 균형점: Precision 68-70%, Recall 40-45%

작성일: 2025-10-10
"""

import os
import sys
import json
import random
import math
from datetime import datetime
from collections import defaultdict

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print(" Phase 4-D Step 2: Threshold 최적화")
print("=" * 80)
print()

# Phase 4-D Step 1의 유틸리티 함수 재사용
def mean(data):
    return sum(data) / len(data) if data else 0

def normal_random(mean_val, std_val):
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mean_val + z * std_val

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

        if self.class_weight == 'balanced':
            n_positive = sum(y)
            n_negative = n_samples - n_positive
            self.weight_positive = n_samples / (2 * n_positive)
            self.weight_negative = n_samples / (2 * n_negative)

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

def calculate_metrics(y_true, y_pred, y_proba):
    """평가 지표 계산"""
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

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

def generate_data(n_samples=15000, accident_rate=0.359):
    """데이터 생성"""
    random.seed(42)
    X = []
    y = []

    for i in range(n_samples):
        is_accident = 1 if random.random() < accident_rate else 0

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

def find_optimal_threshold(y_true, y_proba, optimization='f1', min_precision=0.0, min_recall=0.0):
    """
    최적 Threshold 탐색

    Args:
        y_true: 실제 레이블
        y_proba: 예측 확률
        optimization: 최적화 기준 ('f1', 'precision', 'recall')
        min_precision: 최소 Precision 제약
        min_recall: 최소 Recall 제약

    Returns:
        optimal_threshold, best_score, metrics
    """
    thresholds = [i * 0.01 for i in range(1, 100)]  # 0.01 ~ 0.99
    best_threshold = 0.5
    best_score = 0
    best_metrics = None
    all_results = []

    for threshold in thresholds:
        y_pred = [1 if p >= threshold else 0 for p in y_proba]

        # Confusion matrix
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 제약 조건 확인
        if precision < min_precision or recall < min_recall:
            continue

        # 최적화 기준에 따라 점수 계산
        if optimization == 'f1':
            score = f1
        elif optimization == 'precision':
            score = precision
        elif optimization == 'recall':
            score = recall
        else:
            score = f1

        all_results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
            }

    return best_threshold, best_score, best_metrics, all_results

def main():
    # 1. 데이터 생성
    print("=" * 80)
    print("📊 데이터 생성")
    print("=" * 80)

    X, y = generate_data(n_samples=15000, accident_rate=0.359)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    print(f"\n총 샘플: {len(X):,}개")
    print(f"사고: {sum(y):,}개 ({sum(y)/len(y)*100:.1f}%)")
    print(f"Train: {len(X_train):,}개, Test: {len(X_test):,}개")

    # 2. Class Weight 적용 모델 훈련
    print("\n" + "=" * 80)
    print("🔄 Class Weight 모델 훈련")
    print("=" * 80)

    model = LogisticRegressionWithClassWeight(
        learning_rate=0.01,
        iterations=500,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)

    print("\n모델 훈련 완료")

    results = {}

    # 3. 기본 Threshold (0.5) 성능
    print("\n" + "=" * 80)
    print("📌 Baseline: Threshold = 0.5")
    print("=" * 80)

    y_pred_baseline = model.predict(X_test, threshold=0.5)
    metrics_baseline = calculate_metrics(y_test, y_pred_baseline, y_proba)

    print(f"\n  Precision: {metrics_baseline['precision']:.1%}")
    print(f"  Recall:    {metrics_baseline['recall']:.1%}")
    print(f"  F1 Score:  {metrics_baseline['f1']:.4f}")

    results['baseline_threshold_0.5'] = metrics_baseline

    # 4. F1 Score 최대화 Threshold
    print("\n" + "=" * 80)
    print("✨ 전략 1: F1 Score 최대화")
    print("=" * 80)

    optimal_f1_threshold, best_f1, metrics_f1, all_f1 = find_optimal_threshold(
        y_test, y_proba, optimization='f1'
    )

    print(f"\n  최적 Threshold: {optimal_f1_threshold:.2f}")
    print(f"  Precision: {metrics_f1['precision']:.1%}")
    print(f"  Recall:    {metrics_f1['recall']:.1%}")
    print(f"  F1 Score:  {metrics_f1['f1']:.4f}")

    results['strategy1_f1_max'] = {
        'threshold': optimal_f1_threshold,
        'metrics': metrics_f1
    }

    # 5. Precision ≥ 60% 제약 + Recall 최대화
    print("\n" + "=" * 80)
    print("✨ 전략 2: Precision ≥ 60% 제약 + Recall 최대화")
    print("=" * 80)

    optimal_p60_threshold, best_recall_p60, metrics_p60, all_p60 = find_optimal_threshold(
        y_test, y_proba, optimization='recall', min_precision=0.60
    )

    if metrics_p60:
        print(f"\n  최적 Threshold: {optimal_p60_threshold:.2f}")
        print(f"  Precision: {metrics_p60['precision']:.1%}")
        print(f"  Recall:    {metrics_p60['recall']:.1%}")
        print(f"  F1 Score:  {metrics_p60['f1']:.4f}")

        results['strategy2_precision_60'] = {
            'threshold': optimal_p60_threshold,
            'metrics': metrics_p60
        }
    else:
        print("\n  ⚠️ Precision ≥ 60% 조건을 만족하는 Threshold 없음")

    # 6. Precision ≥ 68% 제약 + Recall 최대화 (목표 균형점)
    print("\n" + "=" * 80)
    print("✨ 전략 3: Precision ≥ 68% 제약 + Recall 최대화 (목표)")
    print("=" * 80)

    optimal_p68_threshold, best_recall_p68, metrics_p68, all_p68 = find_optimal_threshold(
        y_test, y_proba, optimization='recall', min_precision=0.68
    )

    if metrics_p68:
        print(f"\n  최적 Threshold: {optimal_p68_threshold:.2f}")
        print(f"  Precision: {metrics_p68['precision']:.1%}")
        print(f"  Recall:    {metrics_p68['recall']:.1%}")
        print(f"  F1 Score:  {metrics_p68['f1']:.4f}")

        results['strategy3_precision_68'] = {
            'threshold': optimal_p68_threshold,
            'metrics': metrics_p68
        }
    else:
        print("\n  ⚠️ Precision ≥ 68% 조건을 만족하는 Threshold 없음")

    # 7. 균형점 탐색 (Precision 68-70%, Recall 최대화)
    print("\n" + "=" * 80)
    print("🎯 최적 균형점 탐색")
    print("=" * 80)

    best_balance = None
    best_balance_score = 0

    for result in all_f1:
        if 0.68 <= result['precision'] <= 0.70:
            if result['recall'] > best_balance_score:
                best_balance_score = result['recall']
                best_balance = result

    if best_balance:
        print(f"\n  균형점 Threshold: {best_balance['threshold']:.2f}")
        print(f"  Precision: {best_balance['precision']:.1%}")
        print(f"  Recall:    {best_balance['recall']:.1%}")
        print(f"  F1 Score:  {best_balance['f1']:.4f}")

        results['optimal_balance'] = {
            'threshold': best_balance['threshold'],
            'metrics': {
                'precision': best_balance['precision'],
                'recall': best_balance['recall'],
                'f1': best_balance['f1'],
                'confusion_matrix': {
                    'tp': best_balance['tp'],
                    'fp': best_balance['fp'],
                    'tn': best_balance['tn'],
                    'fn': best_balance['fn']
                }
            }
        }
    else:
        print("\n  ⚠️ Precision 68-70% 범위의 Threshold 없음")

    # 8. 비교 분석
    print("\n" + "=" * 80)
    print("📊 전략 비교")
    print("=" * 80)

    print(f"\n{'전략':<30} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)

    print(f"{'Baseline (0.5)':<30} {0.5:>10.2f} {metrics_baseline['precision']:>10.1%} {metrics_baseline['recall']:>10.1%} {metrics_baseline['f1']:>10.4f}")
    print(f"{'F1 최대화':<30} {optimal_f1_threshold:>10.2f} {metrics_f1['precision']:>10.1%} {metrics_f1['recall']:>10.1%} {metrics_f1['f1']:>10.4f}")

    if metrics_p60:
        print(f"{'Precision ≥ 60%':<30} {optimal_p60_threshold:>10.2f} {metrics_p60['precision']:>10.1%} {metrics_p60['recall']:>10.1%} {metrics_p60['f1']:>10.4f}")

    if metrics_p68:
        print(f"{'Precision ≥ 68%':<30} {optimal_p68_threshold:>10.2f} {metrics_p68['precision']:>10.1%} {metrics_p68['recall']:>10.1%} {metrics_p68['f1']:>10.4f}")

    if best_balance:
        print(f"{'균형점 (68-70%)':<30} {best_balance['threshold']:>10.2f} {best_balance['precision']:>10.1%} {best_balance['recall']:>10.1%} {best_balance['f1']:>10.4f}")

    # 9. 결과 저장
    output_file = os.path.join(os.path.dirname(__file__), 'phase4d_step2_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {output_file}")

    # 10. 권장 Threshold
    print("\n" + "=" * 80)
    print("💡 권장사항")
    print("=" * 80)

    print(f"\n✅ 추천 Threshold: {optimal_f1_threshold:.2f} (F1 최대화)")
    print(f"   → Precision: {metrics_f1['precision']:.1%}, Recall: {metrics_f1['recall']:.1%}, F1: {metrics_f1['f1']:.4f}")

    if best_balance:
        print(f"\n🎯 목표 균형점: {best_balance['threshold']:.2f} (Precision 68-70%)")
        print(f"   → Precision: {best_balance['precision']:.1%}, Recall: {best_balance['recall']:.1%}, F1: {best_balance['f1']:.4f}")

    print("\n" + "=" * 80)
    print("✅ Phase 4-D Step 2 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()
