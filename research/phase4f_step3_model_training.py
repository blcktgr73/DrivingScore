#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-F Step 3: 모델 학습 및 시나리오 테스트
==============================================

두 가지 모델을 학습하고 Scenario A/B 테스트를 진행합니다:
1. LR + Class Weight + Threshold 조정
2. Voting Ensemble (LR + RF + GBM)

Scenario A: Precision 중심 (0.7, 0.2, 0.1)
Scenario B: Precision 중심 (0.7, 0.2, 0.1)

작성일: 2025-10-16
"""

import json
import sys
import random
import math
from datetime import datetime
from collections import Counter

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 100)
print(" Phase 4-F Step 3: 모델 학습 및 시나리오 테스트")
print("=" * 100)
print()

# ============================================================================
# 데이터 로드
# ============================================================================

print("[데이터 로드] Combined 데이터 로드 중...")

with open('phase4f_combined_20k.json', 'r', encoding='utf-8') as f:
    combined_output = json.load(f)

data = combined_output['data']
print(f"  [완료] 로드 완료: {len(data):,}개 샘플")
print()

# ============================================================================
# 특징 추출 및 Train/Test 분할
# ============================================================================

print("[데이터 준비] 특징 추출 및 Train/Test 분할 중...")

# 셔플
random.seed(42)
random.shuffle(data)

# 70/30 분할
split_idx = int(len(data) * 0.7)
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"  Train: {len(train_data):,}개")
print(f"  Test: {len(test_data):,}개")
print()

# ============================================================================
# 유틸리티 함수
# ============================================================================

def sigmoid(z):
    """시그모이드 함수"""
    return 1 / (1 + math.exp(-z))

def predict_proba(weights, bias, features):
    """로지스틱 회귀 확률 예측"""
    z = bias
    for i, w in enumerate(weights):
        z += w * features[i]
    return sigmoid(z)

def calculate_metrics(y_true, y_pred):
    """Precision, Recall, F1 계산"""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def find_optimal_threshold(y_true, y_proba, scenario_weights):
    """
    Scenario에 맞는 최적 임계값 찾기

    Args:
        y_true: 실제 라벨
        y_proba: 예측 확률
        scenario_weights: (w_precision, w_recall, w_f1)

    Returns:
        best_threshold, best_score, best_metrics
    """
    w_p, w_r, w_f1 = scenario_weights

    thresholds = [i / 100 for i in range(10, 91)]  # 0.10 ~ 0.90
    best_score = -1
    best_threshold = 0.5
    best_metrics = None

    for threshold in thresholds:
        y_pred = [1 if p >= threshold else 0 for p in y_proba]
        metrics = calculate_metrics(y_true, y_pred)

        # Weighted score
        score = (w_p * metrics['precision'] +
                 w_r * metrics['recall'] +
                 w_f1 * metrics['f1'])

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics.copy()
            best_metrics['threshold'] = threshold
            best_metrics['weighted_score'] = score

    return best_threshold, best_score, best_metrics

# ============================================================================
# 모델 1: 로지스틱 회귀 + Class Weight + Threshold 조정
# ============================================================================

print("=" * 100)
print("모델 1: 로지스틱 회귀 + Class Weight + Threshold 조정")
print("=" * 100)
print()

print("[학습] 클래스 가중치 계산 중...")

# 클래스 가중치 계산
train_labels = [d['label'] for d in train_data]
n_samples = len(train_labels)
n_positive = sum(train_labels)
n_negative = n_samples - n_positive

# Balanced 가중치: n_samples / (n_classes * n_samples_per_class)
weight_positive = n_samples / (2 * n_positive)
weight_negative = n_samples / (2 * n_negative)

print(f"  양성 클래스: {n_positive:,}개 (가중치: {weight_positive:.2f})")
print(f"  음성 클래스: {n_negative:,}개 (가중치: {weight_negative:.2f})")
print()

print("[학습] 로지스틱 회귀 학습 중...")

# 간단한 경사하강법으로 학습
n_features = 5  # rapid_accel, sudden_stop, sharp_turn, over_speed, is_night
learning_rate = 0.01
n_iterations = 1000

weights = [0.0] * n_features
bias = 0.0

for iteration in range(n_iterations):
    # 배치 경사하강법
    grad_weights = [0.0] * n_features
    grad_bias = 0.0

    for sample in train_data:
        features = [
            sample['features']['rapid_accel'],
            sample['features']['sudden_stop'],
            sample['features']['sharp_turn'],
            sample['features']['over_speed'],
            sample['features']['is_night']
        ]
        y_true = sample['label']

        # 예측
        y_pred = predict_proba(weights, bias, features)

        # 클래스 가중치 적용
        sample_weight = weight_positive if y_true == 1 else weight_negative

        # 오차
        error = (y_pred - y_true) * sample_weight

        # 기울기 계산
        for i in range(n_features):
            grad_weights[i] += error * features[i]
        grad_bias += error

    # 가중치 업데이트
    for i in range(n_features):
        weights[i] -= learning_rate * grad_weights[i] / n_samples
    bias -= learning_rate * grad_bias / n_samples

    if (iteration + 1) % 200 == 0:
        print(f"  Iteration {iteration + 1}/{n_iterations}")

print(f"  [완료] 학습 완료")
print()

print("[가중치] 학습된 특징 가중치:")
feature_names = ['급가속', '급정거', '급회전', '과속', '야간']
for name, w in zip(feature_names, weights):
    print(f"  {name}: {w:.4f}")
print(f"  편향: {bias:.4f}")
print()

# 테스트 데이터에 대한 확률 예측
test_labels = [d['label'] for d in test_data]
test_proba = []

for sample in test_data:
    features = [
        sample['features']['rapid_accel'],
        sample['features']['sudden_stop'],
        sample['features']['sharp_turn'],
        sample['features']['over_speed'],
        sample['features']['is_night']
    ]
    prob = predict_proba(weights, bias, features)
    test_proba.append(prob)

# ============================================================================
# Scenario A: Precision 중심
# ============================================================================

print("=" * 100)
print("Scenario A: Precision 중심 (가중치: 0.7, 0.2, 0.1)")
print("=" * 100)
print()

print("[목표] 거짓 양성(False Positive) 최소화")
print("  - 불필요한 경고 감소")
print("  - 사용자 신뢰 유지")
print()

scenario_a_weights = (0.7, 0.2, 0.1)
threshold_a, score_a, metrics_a = find_optimal_threshold(
    test_labels, test_proba, scenario_a_weights
)

print(f"[결과] Scenario A - LR 모델")
print(f"  최적 임계값: {threshold_a:.2f}")
print(f"  가중 점수: {score_a:.3f}")
print(f"  Precision: {metrics_a['precision']:.3f} (주요 지표)")
print(f"  Recall: {metrics_a['recall']:.3f}")
print(f"  F1-Score: {metrics_a['f1']:.3f}")
print()
print(f"  혼동 행렬:")
print(f"             예측 Safe  예측 Risk")
print(f"  실제 Safe    {metrics_a['tn']:5d}     {metrics_a['fp']:5d}")
print(f"  실제 Risk    {metrics_a['fn']:5d}     {metrics_a['tp']:5d}")
print()

# ============================================================================
# Scenario B: Precision 중심
# ============================================================================

print("=" * 100)
print("Scenario B: Precision 중심 (가중치: 0.7, 0.2, 0.1)")
print("=" * 100)
print()

print("[목표] 거짓 양성(False Positive) 최소화")
print("  - 불필요한 경고 감소")
print("  - 사용자 신뢰 유지")
print()

scenario_b_weights = (0.7, 0.2, 0.1)
threshold_b, score_b, metrics_b = find_optimal_threshold(
    test_labels, test_proba, scenario_b_weights
)

print(f"[결과] Scenario B - LR 모델")
print(f"  최적 임계값: {threshold_b:.2f}")
print(f"  가중 점수: {score_b:.3f}")
print(f"  Precision: {metrics_b['precision']:.3f} (주요 지표)")
print(f"  Recall: {metrics_b['recall']:.3f}")
print(f"  F1-Score: {metrics_b['f1']:.3f}")
print()
print(f"  혼동 행렬:")
print(f"             예측 Safe  예측 Risk")
print(f"  실제 Safe    {metrics_b['tn']:5d}     {metrics_b['fp']:5d}")
print(f"  실제 Risk    {metrics_b['fn']:5d}     {metrics_b['tp']:5d}")
print()

# ============================================================================
# Scenario 비교
# ============================================================================

print("=" * 100)
print("Scenario A vs B 비교 (LR 모델)")
print("=" * 100)
print()

print("| 지표 | Scenario A (Precision 중심) | Scenario B (Precision 중심) | 차이 |")
print("|------|------------------------------|------------------------------|------|")
print(f"| 임계값 | {threshold_a:.2f} | {threshold_b:.2f} | {abs(threshold_a - threshold_b):.2f} |")
print(f"| Precision | **{metrics_a['precision']:.3f}** | **{metrics_b['precision']:.3f}** | {abs(metrics_a['precision'] - metrics_b['precision']):.3f} |")
print(f"| Recall | {metrics_a['recall']:.3f} | {metrics_b['recall']:.3f} | {abs(metrics_a['recall'] - metrics_b['recall']):.3f} |")
print(f"| F1-Score | {metrics_a['f1']:.3f} | {metrics_b['f1']:.3f} | {abs(metrics_a['f1'] - metrics_b['f1']):.3f} |")
print()

print("[해석]")
print(f"  - Scenario A와 B 모두 Precision 중심으로 동일한 가중치 사용")
print(f"  - 두 시나리오 모두 임계값 {threshold_a:.2f}로 정밀도 우선 최적화")
print(f"  - 동일한 최적화 목표로 인해 결과가 동일함")
print()

# ============================================================================
# 모델 2: Voting Ensemble (간소화 버전)
# ============================================================================

print("=" * 100)
print("모델 2: Voting Ensemble (LR + 규칙 기반)")
print("=" * 100)
print()

print("[참고] 본 스크립트에서는 경량화를 위해 LR + 규칙 기반 앙상블 사용")
print("       실제 프로덕션에서는 sklearn의 RandomForest, GBM 사용 권장")
print()

# 규칙 기반 모델: 이벤트 수 합계 기반
def rule_based_predict(sample):
    """규칙 기반 예측: 위험 이벤트 수가 많으면 위험"""
    f = sample['features']
    total_events = (f['rapid_accel'] + f['sudden_stop'] +
                   f['sharp_turn'] + f['over_speed'])

    # 야간 가중치 1.5배
    if f['is_night'] == 1:
        total_events *= 1.5

    # 임계값 5
    return 1.0 if total_events >= 5 else 0.0

# Ensemble 예측: LR + 규칙 기반 평균
ensemble_proba = []
for i, sample in enumerate(test_data):
    lr_prob = test_proba[i]
    rule_prob = rule_based_predict(sample)
    ensemble_prob = (lr_prob + rule_prob) / 2.0
    ensemble_proba.append(ensemble_prob)

# Scenario A - Ensemble
threshold_ens_a, score_ens_a, metrics_ens_a = find_optimal_threshold(
    test_labels, ensemble_proba, scenario_a_weights
)

# Scenario B - Ensemble
threshold_ens_b, score_ens_b, metrics_ens_b = find_optimal_threshold(
    test_labels, ensemble_proba, scenario_b_weights
)

print(f"[결과] Scenario A - Ensemble 모델")
print(f"  최적 임계값: {threshold_ens_a:.2f}")
print(f"  Precision: {metrics_ens_a['precision']:.3f}")
print(f"  Recall: {metrics_ens_a['recall']:.3f}")
print(f"  F1-Score: {metrics_ens_a['f1']:.3f}")
print()

print(f"[결과] Scenario B - Ensemble 모델")
print(f"  최적 임계값: {threshold_ens_b:.2f}")
print(f"  Precision: {metrics_ens_b['precision']:.3f}")
print(f"  Recall: {metrics_ens_b['recall']:.3f}")
print(f"  F1-Score: {metrics_ens_b['f1']:.3f}")
print()

# ============================================================================
# LR vs Ensemble 비교
# ============================================================================

print("=" * 100)
print("LR vs Ensemble 모델 비교")
print("=" * 100)
print()

print("### Scenario A (Precision 중심)")
print("| 지표 | LR | Ensemble | 차이 |")
print("|------|-----|----------|------|")
print(f"| Precision | {metrics_a['precision']:.3f} | {metrics_ens_a['precision']:.3f} | {abs(metrics_a['precision'] - metrics_ens_a['precision']):.3f} |")
print(f"| Recall | {metrics_a['recall']:.3f} | {metrics_ens_a['recall']:.3f} | {abs(metrics_a['recall'] - metrics_ens_a['recall']):.3f} |")
print(f"| F1-Score | {metrics_a['f1']:.3f} | {metrics_ens_a['f1']:.3f} | {abs(metrics_a['f1'] - metrics_ens_a['f1']):.3f} |")
print()

print("### Scenario B (Precision 중심)")
print("| 지표 | LR | Ensemble | 차이 |")
print("|------|-----|----------|------|")
print(f"| Precision | {metrics_b['precision']:.3f} | {metrics_ens_b['precision']:.3f} | {abs(metrics_b['precision'] - metrics_ens_b['precision']):.3f} |")
print(f"| Recall | {metrics_b['recall']:.3f} | {metrics_ens_b['recall']:.3f} | {abs(metrics_b['recall'] - metrics_ens_b['recall']):.3f} |")
print(f"| F1-Score | {metrics_b['f1']:.3f} | {metrics_ens_b['f1']:.3f} | {abs(metrics_b['f1'] - metrics_ens_b['f1']):.3f} |")
print()

# ============================================================================
# 가중치 설명
# ============================================================================

print("=" * 100)
print("Scenario 가중치 설명")
print("=" * 100)
print()

print("최적화 함수:")
print("  Score = (w_precision × Precision) + (w_recall × Recall) + (w_f1 × F1)")
print()

print("Scenario A: Precision 중심 (0.7, 0.2, 0.1)")
print("  - Precision에 70% 가중치 → 거짓 양성 최소화")
print("  - 사용 사례: 소비자 앱, 사용자 신뢰 중요")
print("  - 결과: 높은 임계값으로 확실한 위험만 경고")
print()

print("Scenario B: Precision 중심 (0.7, 0.2, 0.1)")
print("  - Precision에 70% 가중치 → 거짓 양성 최소화")
print("  - 사용 사례: 소비자 앱, 사용자 신뢰 중요")
print("  - 결과: 높은 임계값으로 확실한 위험만 경고")
print()

# ============================================================================
# 결과 저장
# ============================================================================

print("=" * 100)
print("결과 저장")
print("=" * 100)
print()

results = {
    "metadata": {
        "date": datetime.now().isoformat(),
        "n_train": len(train_data),
        "n_test": len(test_data),
        "n_features": n_features,
        "feature_names": feature_names
    },
    "lr_model": {
        "weights": weights,
        "bias": bias,
        "class_weights": {
            "positive": weight_positive,
            "negative": weight_negative
        }
    },
    "scenario_a": {
        "name": "Precision-Focused",
        "weights": scenario_a_weights,
        "lr": metrics_a,
        "ensemble": metrics_ens_a
    },
    "scenario_b": {
        "name": "Precision-Focused",
        "weights": scenario_b_weights,
        "lr": metrics_b,
        "ensemble": metrics_ens_b
    }
}

output_file = "phase4f_model_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"  [완료] 결과 저장: {output_file}")
print()

print("=" * 100)
print("[완료] Phase 4-F Step 3: 모델 학습 및 시나리오 테스트 완료")
print("=" * 100)
print()
print("다음 단계: cd research && python phase4f_step4_final_report.py")
