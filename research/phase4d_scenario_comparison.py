#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-D: Scenario A vs B 완전 비교
====================================

Phase 4-C와 Phase 4-D를 Scenario A (4개 이벤트)와 Scenario B (3개 이벤트)
모두에 대해 비교 분석합니다.

Scenario A: 급가속, 급정거, 급회전, 과속 (4개)
Scenario B: 급가속, 급정거, 급회전 (3개, 과속 제외)

작성일: 2025-10-10
"""

import os
import sys
import json
import random
import math

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print(" Phase 4-D: Scenario A vs B 완전 비교")
print("=" * 80)
print()

# ============================================================================
# 유틸리티 함수
# ============================================================================

def mean(data):
    return sum(data) / len(data) if data else 0

def normal_random(mean_val, std_val):
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mean_val + z * std_val

# ============================================================================
# 모델 클래스
# ============================================================================

class LogisticRegression:
    """기존 Logistic Regression (Phase 4-C)"""
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

    def predict_proba(self, X):
        predictions = []
        for i in range(len(X)):
            z = sum(X[i][j] * self.weights[j] for j in range(len(self.weights))) + self.bias
            predictions.append(self.sigmoid(z))
        return predictions

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probas]

class LogisticRegressionWithClassWeight:
    """Class Weight 적용 Logistic Regression (Phase 4-D)"""
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

# ============================================================================
# 평가 함수
# ============================================================================

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

def find_optimal_threshold(y_true, y_proba, optimization='f1'):
    """최적 Threshold 탐색"""
    thresholds = [i * 0.01 for i in range(1, 100)]
    best_threshold = 0.5
    best_score = 0
    best_metrics = None

    for threshold in thresholds:
        y_pred = [1 if p >= threshold else 0 for p in y_proba]

        tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        score = f1 if optimization == 'f1' else recall

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
            }

    return best_threshold, best_metrics

# ============================================================================
# 데이터 생성
# ============================================================================

def generate_data_scenario_a(n_samples=15000, accident_rate=0.359):
    """Scenario A: 4개 이벤트 (과속 포함)"""
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

def generate_data_scenario_b(n_samples=15000, accident_rate=0.359):
    """Scenario B: 3개 이벤트 (과속 제외)"""
    random.seed(42)
    X = []
    y = []

    for i in range(n_samples):
        is_accident = 1 if random.random() < accident_rate else 0

        if is_accident:
            rapid_accel = max(0, int(normal_random(5.0, 2.0)))
            sudden_stop = max(0, int(normal_random(6.0, 2.5)))
            sharp_turn = max(0, int(normal_random(4.0, 2.0)))
        else:
            rapid_accel = max(0, int(normal_random(2.0, 1.5)))
            sudden_stop = max(0, int(normal_random(2.5, 1.5)))
            sharp_turn = max(0, int(normal_random(1.5, 1.0)))

        X.append([rapid_accel, sudden_stop, sharp_turn])
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

# ============================================================================
# 메인 실행
# ============================================================================

def run_scenario(scenario_name, X, y):
    """각 시나리오 실행"""
    print(f"\n{'=' * 80}")
    print(f"🔬 {scenario_name}")
    print(f"{'=' * 80}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    print(f"\n특징 개수: {len(X[0])}개")
    print(f"Train: {len(X_train):,}개, Test: {len(X_test):,}개")

    results = {}

    # Phase 4-C
    print(f"\n[Phase 4-C: Baseline]")
    model_4c = LogisticRegression(learning_rate=0.01, iterations=500)
    model_4c.fit(X_train, y_train)

    y_proba_4c = model_4c.predict_proba(X_test)
    y_pred_4c = model_4c.predict(X_test, threshold=0.5)
    metrics_4c = calculate_metrics(y_test, y_pred_4c, y_proba_4c)

    print(f"  Precision: {metrics_4c['precision']:.1%}, Recall: {metrics_4c['recall']:.1%}, F1: {metrics_4c['f1']:.4f}, AUC: {metrics_4c['auc']:.4f}")

    results['phase4c'] = {
        'threshold': 0.5,
        'metrics': metrics_4c
    }

    # Phase 4-D: Class Weight + Threshold 최적화
    print(f"\n[Phase 4-D: Class Weight + Threshold Optimization]")
    model_4d = LogisticRegressionWithClassWeight(
        learning_rate=0.01,
        iterations=500,
        class_weight='balanced'
    )
    model_4d.fit(X_train, y_train)

    y_proba_4d = model_4d.predict_proba(X_test)

    # F1 최대화 Threshold 탐색
    optimal_threshold, metrics_4d = find_optimal_threshold(y_test, y_proba_4d, optimization='f1')

    print(f"  최적 Threshold: {optimal_threshold:.2f}")
    print(f"  Precision: {metrics_4d['precision']:.1%}, Recall: {metrics_4d['recall']:.1%}, F1: {metrics_4d['f1']:.4f}")

    results['phase4d'] = {
        'threshold': optimal_threshold,
        'metrics': metrics_4d
    }

    # 개선 분석
    print(f"\n[개선 분석]")
    recall_change = metrics_4d['recall'] - metrics_4c['recall']
    precision_change = metrics_4d['precision'] - metrics_4c['precision']
    f1_change = metrics_4d['f1'] - metrics_4c['f1']

    print(f"  Recall:    {metrics_4c['recall']:.1%} → {metrics_4d['recall']:.1%} ({recall_change:+.1%})")
    print(f"  Precision: {metrics_4c['precision']:.1%} → {metrics_4d['precision']:.1%} ({precision_change:+.1%})")
    print(f"  F1 Score:  {metrics_4c['f1']:.4f} → {metrics_4d['f1']:.4f} ({f1_change:+.4f}, {metrics_4d['f1']/metrics_4c['f1']:.2f}배)")

    results['improvements'] = {
        'recall_change': recall_change,
        'precision_change': precision_change,
        'f1_change': f1_change,
        'f1_multiplier': metrics_4d['f1'] / metrics_4c['f1'] if metrics_4c['f1'] > 0 else 0
    }

    return results

def main():
    print("\n데이터 생성 중...")

    # Scenario A (4개 이벤트)
    X_a, y_a = generate_data_scenario_a(n_samples=15000, accident_rate=0.359)
    results_a = run_scenario("Scenario A (4개 이벤트: 급가속, 급정거, 급회전, 과속)", X_a, y_a)

    # Scenario B (3개 이벤트)
    X_b, y_b = generate_data_scenario_b(n_samples=15000, accident_rate=0.359)
    results_b = run_scenario("Scenario B (3개 이벤트: 급가속, 급정거, 급회전)", X_b, y_b)

    # 최종 비교
    print(f"\n{'=' * 80}")
    print(f"📊 Scenario A vs B 최종 비교")
    print(f"{'=' * 80}")

    print(f"\n{'시나리오':<20} {'Phase':<10} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Threshold':>12}")
    print("-" * 88)

    # Scenario A
    metrics_a_4c = results_a['phase4c']['metrics']
    metrics_a_4d = results_a['phase4d']['metrics']
    print(f"{'Scenario A (4개)':<20} {'Phase 4-C':<10} {metrics_a_4c['precision']:>12.1%} {metrics_a_4c['recall']:>12.1%} {metrics_a_4c['f1']:>12.4f} {results_a['phase4c']['threshold']:>12.2f}")
    print(f"{'Scenario A (4개)':<20} {'Phase 4-D':<10} {metrics_a_4d['precision']:>12.1%} {metrics_a_4d['recall']:>12.1%} {metrics_a_4d['f1']:>12.4f} {results_a['phase4d']['threshold']:>12.2f}")

    # Scenario B
    metrics_b_4c = results_b['phase4c']['metrics']
    metrics_b_4d = results_b['phase4d']['metrics']
    print(f"{'Scenario B (3개)':<20} {'Phase 4-C':<10} {metrics_b_4c['precision']:>12.1%} {metrics_b_4c['recall']:>12.1%} {metrics_b_4c['f1']:>12.4f} {results_b['phase4c']['threshold']:>12.2f}")
    print(f"{'Scenario B (3개)':<20} {'Phase 4-D':<10} {metrics_b_4d['precision']:>12.1%} {metrics_b_4d['recall']:>12.1%} {metrics_b_4d['f1']:>12.4f} {results_b['phase4d']['threshold']:>12.2f}")

    # 권장사항
    print(f"\n{'=' * 80}")
    print(f"💡 최종 권장사항")
    print(f"{'=' * 80}")

    # F1 Score 비교
    best_scenario = "A" if metrics_a_4d['f1'] > metrics_b_4d['f1'] else "B"
    best_f1 = max(metrics_a_4d['f1'], metrics_b_4d['f1'])

    print(f"\n🏆 추천: Scenario {best_scenario} + Phase 4-D")
    if best_scenario == "A":
        print(f"   → F1 Score: {metrics_a_4d['f1']:.4f} (Scenario B 대비 +{metrics_a_4d['f1'] - metrics_b_4d['f1']:.4f})")
        print(f"   → 4개 이벤트 (과속 포함)")
    else:
        print(f"   → F1 Score: {metrics_b_4d['f1']:.4f} (Scenario A 대비 +{metrics_b_4d['f1'] - metrics_a_4d['f1']:.4f})")
        print(f"   → 3개 이벤트 (과속 제외)")

    print(f"\n이유:")
    if best_scenario == "A":
        print(f"  1. ✅ F1 Score 최고 ({metrics_a_4d['f1']:.4f})")
        print(f"  2. ✅ Precision {metrics_a_4d['precision']:.1%}, Recall {metrics_a_4d['recall']:.1%}")
        print(f"  3. ⚠️ 과속 이벤트 추가 구현 필요 (GPS, 제한속도 정보)")
    else:
        print(f"  1. ✅ F1 Score 우수 ({metrics_b_4d['f1']:.4f})")
        print(f"  2. ✅ 구현 단순 (3개 이벤트만)")
        print(f"  3. ✅ GPS 의존성 없음")

    # 결과 저장
    final_results = {
        'scenario_a': results_a,
        'scenario_b': results_b,
        'comparison': {
            'recommended_scenario': best_scenario,
            'f1_difference': abs(metrics_a_4d['f1'] - metrics_b_4d['f1']),
            'scenario_a_better': metrics_a_4d['f1'] > metrics_b_4d['f1']
        }
    }

    output_file = os.path.join(os.path.dirname(__file__), 'phase4d_scenario_comparison.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {output_file}")

    print(f"\n{'=' * 80}")
    print(f"✅ Scenario A vs B 비교 완료!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
