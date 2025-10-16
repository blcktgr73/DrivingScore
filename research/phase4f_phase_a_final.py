#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-F Phase A: 최종 상품화 수준 모델 개선
===============================================

Scenario 재정의:
- Scenario A: 4개 이벤트 (급가속, 급정거, 급회전, 과속 포함)
- Scenario B: 3개 이벤트 (급가속, 급정거, 급회전만, 과속 제외)

목표:
1. Recall 우선 최적화 (행동 변화 유도)
2. Linear Scoring 가중치 도출 (Day/Night 구분)
3. Scenario A vs B 비교 분석

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
print(" Phase 4-F Phase A: 최종 상품화 수준 모델 개선")
print(" Scenario A (4개 이벤트) vs Scenario B (3개 이벤트)")
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
# Scenario 정의
# ============================================================================

print("[Scenario 정의]")
print()
print("Scenario A: 4개 이벤트 포함")
print("  - 급가속 (rapid_accel)")
print("  - 급정거 (sudden_stop)")
print("  - 급회전 (sharp_turn)")
print("  - 과속 (over_speed) ★")
print()
print("Scenario B: 3개 이벤트만 (과속 제외)")
print("  - 급가속 (rapid_accel)")
print("  - 급정거 (sudden_stop)")
print("  - 급회전 (sharp_turn)")
print("  - 야간 (is_night)")
print()

# ============================================================================
# 특징 엔지니어링 (Scenario별)
# ============================================================================

print("[특징 엔지니어링] Scenario별 특징 생성 중...")

def add_scenario_a_features(sample):
    """Scenario A: 4개 이벤트 + 엔지니어링"""
    f = sample['features']

    # 기본 4개 이벤트 총합
    total = f['rapid_accel'] + f['sudden_stop'] + f['sharp_turn'] + f['over_speed']

    # 추가 특징
    sample['features_a'] = {
        'rapid_accel': f['rapid_accel'],
        'sudden_stop': f['sudden_stop'],
        'sharp_turn': f['sharp_turn'],
        'over_speed': f['over_speed'],
        'is_night': f['is_night'],
        'total_events': total,
        'risky_ratio': (f['rapid_accel'] + f['sudden_stop']) / max(total, 1),
        'night_risky': (f['rapid_accel'] + f['sudden_stop']) * f['is_night'] * 1.5,
        'emergency': min(f['rapid_accel'], f['sudden_stop']),
        'overspeed_turn': f['over_speed'] * f['sharp_turn'],
        'event_density': total / max(sample['metadata'].get('trip_duration', 60), 1)
    }

    return sample

def add_scenario_b_features(sample):
    """Scenario B: 3개 이벤트 (과속 제외) + 엔지니어링"""
    f = sample['features']

    # 3개 이벤트 총합 (과속 제외)
    total = f['rapid_accel'] + f['sudden_stop'] + f['sharp_turn']

    # 추가 특징
    sample['features_b'] = {
        'rapid_accel': f['rapid_accel'],
        'sudden_stop': f['sudden_stop'],
        'sharp_turn': f['sharp_turn'],
        'is_night': f['is_night'],
        'total_events': total,
        'risky_ratio': (f['rapid_accel'] + f['sudden_stop']) / max(total, 1),
        'night_risky': (f['rapid_accel'] + f['sudden_stop']) * f['is_night'] * 1.5,
        'emergency': min(f['rapid_accel'], f['sudden_stop']),
        'event_density': total / max(sample['metadata'].get('trip_duration', 60), 1)
    }

    return sample

# 모든 샘플에 적용
for i, sample in enumerate(data):
    data[i] = add_scenario_a_features(sample)
    data[i] = add_scenario_b_features(data[i])

print("  [완료] Scenario A: 11개 특징")
print("  [완료] Scenario B: 9개 특징 (과속 관련 2개 제외)")
print()

# ============================================================================
# Train/Test 분할
# ============================================================================

print("[데이터 준비] Train/Test 분할 중...")

random.seed(42)
random.shuffle(data)

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
    return 1 / (1 + math.exp(-max(min(z, 500), -500)))  # overflow 방지

def predict_proba(weights, bias, features):
    """로지스틱 회귀 확률 예측"""
    z = bias
    for i, w in enumerate(weights):
        z += w * features[i]
    return sigmoid(z)

def calculate_metrics(y_true, y_pred):
    """Precision, Recall, F1, Accuracy 계산"""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def calculate_auc(y_true, y_proba):
    """AUC 계산"""
    pairs = sorted(zip(y_proba, y_true), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc_sum = 0
    for prob, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc_sum += tp
    return auc_sum / (n_pos * n_neg)

def find_optimal_threshold(y_true, y_proba, scenario_weights):
    """Recall 중심 최적 임계값 찾기"""
    w_p, w_r, w_f1 = scenario_weights
    thresholds = [i / 100 for i in range(5, 91)]
    best_score = -1
    best_threshold = 0.5
    best_metrics = None

    for threshold in thresholds:
        y_pred = [1 if p >= threshold else 0 for p in y_proba]
        metrics = calculate_metrics(y_true, y_pred)
        score = (w_p * metrics['precision'] + w_r * metrics['recall'] + w_f1 * metrics['f1'])

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics.copy()
            best_metrics['threshold'] = threshold
            best_metrics['weighted_score'] = score

    return best_threshold, best_score, best_metrics

def train_logistic_regression(train_data, feature_key, feature_names, weight_multiplier=2.5):
    """로지스틱 회귀 학습"""
    train_labels = [d['label'] for d in train_data]
    n_samples = len(train_labels)
    n_positive = sum(train_labels)
    n_negative = n_samples - n_positive

    weight_positive = (n_samples / (2 * n_positive)) * weight_multiplier
    weight_negative = n_samples / (2 * n_negative)

    n_features = len(feature_names)
    learning_rate = 0.01
    n_iterations = 1000
    weights = [0.0] * n_features
    bias = 0.0

    for iteration in range(n_iterations):
        grad_weights = [0.0] * n_features
        grad_bias = 0.0

        for sample in train_data:
            features = [sample[feature_key][name] for name in feature_names]
            y_true = sample['label']
            y_pred = predict_proba(weights, bias, features)
            sample_weight = weight_positive if y_true == 1 else weight_negative
            error = (y_pred - y_true) * sample_weight

            for i in range(n_features):
                grad_weights[i] += error * features[i]
            grad_bias += error

        for i in range(n_features):
            weights[i] -= learning_rate * grad_weights[i] / n_samples
        bias -= learning_rate * grad_bias / n_samples

    return weights, bias, weight_positive, weight_negative

# ============================================================================
# Scenario A 학습 (4개 이벤트)
# ============================================================================

print("=" * 100)
print("Scenario A: 4개 이벤트 모델 학습")
print("=" * 100)
print()

feature_names_a = [
    'rapid_accel', 'sudden_stop', 'sharp_turn', 'over_speed', 'is_night',
    'total_events', 'risky_ratio', 'night_risky', 'emergency', 'overspeed_turn', 'event_density'
]

print("[학습] Scenario A 로지스틱 회귀 학습 중...")
weights_a, bias_a, weight_pos_a, weight_neg_a = train_logistic_regression(
    train_data, 'features_a', feature_names_a
)
print("  [완료] 학습 완료")
print()

print("[가중치] Scenario A 학습된 특징 가중치:")
feature_names_kr_a = [
    '급가속', '급정거', '급회전', '과속', '야간',
    '이벤트총합', '위험비율', '야간위험', '긴급상황', '과속회전', '이벤트밀도'
]
for name_kr, name_en, w in zip(feature_names_kr_a, feature_names_a, weights_a):
    print(f"  {name_kr:12s} ({name_en:15s}): {w:+.4f}")
print(f"  편향 (bias): {bias_a:+.4f}")
print()

# 테스트 예측
test_labels = [d['label'] for d in test_data]
test_proba_a = []
for sample in test_data:
    features = [sample['features_a'][name] for name in feature_names_a]
    prob = predict_proba(weights_a, bias_a, features)
    test_proba_a.append(prob)

auc_a = calculate_auc(test_labels, test_proba_a)
print(f"[AUC] Scenario A Test AUC: {auc_a:.4f}")
print()

# Recall 중심 최적화
scenario_weights = (0.2, 0.7, 0.1)  # Recall 중심
threshold_a, score_a, metrics_a = find_optimal_threshold(test_labels, test_proba_a, scenario_weights)

print(f"[결과] Scenario A (Recall 중심)")
print(f"  최적 임계값: {threshold_a:.2f}")
print(f"  Precision: {metrics_a['precision']:.3f}")
print(f"  Recall: {metrics_a['recall']:.3f}")
print(f"  F1-Score: {metrics_a['f1']:.3f}")
print(f"  Accuracy: {metrics_a['accuracy']:.3f}")
print()

# ============================================================================
# Scenario B 학습 (3개 이벤트, 과속 제외)
# ============================================================================

print("=" * 100)
print("Scenario B: 3개 이벤트 모델 학습 (과속 제외)")
print("=" * 100)
print()

feature_names_b = [
    'rapid_accel', 'sudden_stop', 'sharp_turn', 'is_night',
    'total_events', 'risky_ratio', 'night_risky', 'emergency', 'event_density'
]

print("[학습] Scenario B 로지스틱 회귀 학습 중...")
weights_b, bias_b, weight_pos_b, weight_neg_b = train_logistic_regression(
    train_data, 'features_b', feature_names_b
)
print("  [완료] 학습 완료")
print()

print("[가중치] Scenario B 학습된 특징 가중치:")
feature_names_kr_b = [
    '급가속', '급정거', '급회전', '야간',
    '이벤트총합', '위험비율', '야간위험', '긴급상황', '이벤트밀도'
]
for name_kr, name_en, w in zip(feature_names_kr_b, feature_names_b, weights_b):
    print(f"  {name_kr:12s} ({name_en:15s}): {w:+.4f}")
print(f"  편향 (bias): {bias_b:+.4f}")
print()

# 테스트 예측
test_proba_b = []
for sample in test_data:
    features = [sample['features_b'][name] for name in feature_names_b]
    prob = predict_proba(weights_b, bias_b, features)
    test_proba_b.append(prob)

auc_b = calculate_auc(test_labels, test_proba_b)
print(f"[AUC] Scenario B Test AUC: {auc_b:.4f}")
print()

# Recall 중심 최적화
threshold_b, score_b, metrics_b = find_optimal_threshold(test_labels, test_proba_b, scenario_weights)

print(f"[결과] Scenario B (Recall 중심)")
print(f"  최적 임계값: {threshold_b:.2f}")
print(f"  Precision: {metrics_b['precision']:.3f}")
print(f"  Recall: {metrics_b['recall']:.3f}")
print(f"  F1-Score: {metrics_b['f1']:.3f}")
print(f"  Accuracy: {metrics_b['accuracy']:.3f}")
print()

# ============================================================================
# Scenario A vs B 비교
# ============================================================================

print("=" * 100)
print("Scenario A (4개) vs B (3개) 비교")
print("=" * 100)
print()

print("| 지표 | Scenario A (4개 이벤트) | Scenario B (3개 이벤트) | 차이 |")
print("|------|------------------------|------------------------|------|")
print(f"| 특징 수 | 11개 | 9개 | -2개 (과속 관련) |")
print(f"| AUC | {auc_a:.4f} | {auc_b:.4f} | {auc_a - auc_b:+.4f} |")
print(f"| 임계값 | {threshold_a:.2f} | {threshold_b:.2f} | {abs(threshold_a - threshold_b):.2f} |")
print(f"| Precision | {metrics_a['precision']:.3f} | {metrics_b['precision']:.3f} | {metrics_a['precision'] - metrics_b['precision']:+.3f} |")
print(f"| Recall | {metrics_a['recall']:.3f} | {metrics_b['recall']:.3f} | {metrics_a['recall'] - metrics_b['recall']:+.3f} |")
print(f"| F1-Score | {metrics_a['f1']:.3f} | {metrics_b['f1']:.3f} | {metrics_a['f1'] - metrics_b['f1']:+.3f} |")
print(f"| Accuracy | {metrics_a['accuracy']:.3f} | {metrics_b['accuracy']:.3f} | {metrics_a['accuracy'] - metrics_b['accuracy']:+.3f} |")
print()

print("[해석]")
if metrics_a['f1'] > metrics_b['f1']:
    print(f"  ✅ Scenario A가 F1-Score {(metrics_a['f1'] - metrics_b['f1'])*100:.1f}%p 우세")
    print(f"  → 과속 이벤트가 모델 성능 향상에 기여")
else:
    print(f"  ✅ Scenario B가 F1-Score {(metrics_b['f1'] - metrics_a['f1'])*100:.1f}%p 우세")
    print(f"  → 과속 제외해도 성능 유지 (구현 단순화 가능)")
print()

# ============================================================================
# Linear Scoring 가중치 도출
# ============================================================================

print("=" * 100)
print("Linear Scoring 가중치 도출 (Day/Night 구분)")
print("=" * 100)
print()

def derive_linear_weights(weights, feature_names, basic_events):
    """로지스틱 가중치를 Linear Scoring 가중치로 변환"""
    base_weights = {}
    for event in basic_events:
        idx = feature_names.index(event)
        base_weights[event] = abs(weights[idx])

    # 정규화 (1~5점 범위)
    max_weight = max(base_weights.values())
    scaling_factor = 5.0 / max_weight if max_weight > 0 else 1.0

    day_penalties = {}
    for event, weight in base_weights.items():
        day_penalties[event] = weight * scaling_factor

    # Night 1.5배 가중
    night_penalties = {event: penalty * 1.5 for event, penalty in day_penalties.items()}

    return day_penalties, night_penalties

# Scenario A Linear 가중치
basic_events_a = ['rapid_accel', 'sudden_stop', 'sharp_turn', 'over_speed']
day_penalties_a, night_penalties_a = derive_linear_weights(weights_a, feature_names_a, basic_events_a)

# Scenario B Linear 가중치
basic_events_b = ['rapid_accel', 'sudden_stop', 'sharp_turn']
day_penalties_b, night_penalties_b = derive_linear_weights(weights_b, feature_names_b, basic_events_b)

event_names_kr = {
    'rapid_accel': '급가속',
    'sudden_stop': '급정거',
    'sharp_turn': '급회전',
    'over_speed': '과속'
}

print("### Scenario A: Day 감점 가중치 (이벤트 1회당)")
print()
print("| 이벤트 | 감점 (점) | 로지스틱 가중치 |")
print("|--------|-----------|----------------|")
for event in basic_events_a:
    print(f"| {event_names_kr[event]} | {day_penalties_a[event]:.2f}점 | {weights_a[feature_names_a.index(event)]:+.4f} |")
print()

print("### Scenario A: Night 감점 가중치 (1.5배)")
print()
print("| 이벤트 | 감점 (점) | Day 대비 |")
print("|--------|-----------|----------|")
for event in basic_events_a:
    print(f"| {event_names_kr[event]} | {night_penalties_a[event]:.2f}점 | 1.5배 |")
print()

print("### Scenario B: Day 감점 가중치 (이벤트 1회당)")
print()
print("| 이벤트 | 감점 (점) | 로지스틱 가중치 |")
print("|--------|-----------|----------------|")
for event in basic_events_b:
    print(f"| {event_names_kr[event]} | {day_penalties_b[event]:.2f}점 | {weights_b[feature_names_b.index(event)]:+.4f} |")
print()

print("### Scenario B: Night 감점 가중치 (1.5배)")
print()
print("| 이벤트 | 감점 (점) | Day 대비 |")
print("|--------|-----------|----------|")
for event in basic_events_b:
    print(f"| {event_names_kr[event]} | {night_penalties_b[event]:.2f}점 | 1.5배 |")
print()

# ============================================================================
# 점수 계산 예시
# ============================================================================

print("=" * 100)
print("점수 계산 예시 (100점 만점)")
print("=" * 100)
print()

# Risk/Safe Group 평균 이벤트
risk_events = {'급가속': 2.79, '급정거': 2.26, '급회전': 1.90, '과속': 1.38}
safe_events = {'급가속': 0.87, '급정거': 0.83, '급회전': 0.74, '과속': 0.51}

event_map = {
    '급가속': 'rapid_accel',
    '급정거': 'sudden_stop',
    '급회전': 'sharp_turn',
    '과속': 'over_speed'
}

# Scenario A 점수
print("### Scenario A (4개 이벤트)")
print()
print("예시 1: Risk Group 평균 (Day)")
deduction_a_risk_day = sum(risk_events[kr] * day_penalties_a[event_map[kr]] for kr in risk_events)
score_a_risk_day = max(0, 100 - deduction_a_risk_day)
print(f"  감점: {deduction_a_risk_day:.2f}점, 최종 점수: {score_a_risk_day:.1f}점")
print()

print("예시 2: Risk Group 평균 (Night)")
deduction_a_risk_night = sum(risk_events[kr] * night_penalties_a[event_map[kr]] for kr in risk_events)
score_a_risk_night = max(0, 100 - deduction_a_risk_night)
print(f"  감점: {deduction_a_risk_night:.2f}점, 최종 점수: {score_a_risk_night:.1f}점")
print()

print("예시 3: Safe Group 평균 (Day)")
deduction_a_safe_day = sum(safe_events[kr] * day_penalties_a[event_map[kr]] for kr in safe_events)
score_a_safe_day = max(0, 100 - deduction_a_safe_day)
print(f"  감점: {deduction_a_safe_day:.2f}점, 최종 점수: {score_a_safe_day:.1f}점")
print()

# Scenario B 점수 (과속 제외)
print("### Scenario B (3개 이벤트, 과속 제외)")
print()
risk_events_b = {k: v for k, v in risk_events.items() if k != '과속'}
safe_events_b = {k: v for k, v in safe_events.items() if k != '과속'}

print("예시 1: Risk Group 평균 (Day)")
deduction_b_risk_day = sum(risk_events_b[kr] * day_penalties_b[event_map[kr]] for kr in risk_events_b)
score_b_risk_day = max(0, 100 - deduction_b_risk_day)
print(f"  감점: {deduction_b_risk_day:.2f}점, 최종 점수: {score_b_risk_day:.1f}점")
print()

print("예시 2: Risk Group 평균 (Night)")
deduction_b_risk_night = sum(risk_events_b[kr] * night_penalties_b[event_map[kr]] for kr in risk_events_b)
score_b_risk_night = max(0, 100 - deduction_b_risk_night)
print(f"  감점: {deduction_b_risk_night:.2f}점, 최종 점수: {score_b_risk_night:.1f}점")
print()

print("예시 3: Safe Group 평균 (Day)")
deduction_b_safe_day = sum(safe_events_b[kr] * day_penalties_b[event_map[kr]] for kr in safe_events_b)
score_b_safe_day = max(0, 100 - deduction_b_safe_day)
print(f"  감점: {deduction_b_safe_day:.2f}점, 최종 점수: {score_b_safe_day:.1f}점")
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
        "phase": "Phase A Final",
        "n_train": len(train_data),
        "n_test": len(test_data)
    },
    "scenario_a": {
        "name": "4 Events (급가속, 급정거, 급회전, 과속)",
        "n_features": len(feature_names_a),
        "feature_names": feature_names_a,
        "weights": weights_a,
        "bias": bias_a,
        "class_weight_positive": weight_pos_a,
        "metrics": metrics_a,
        "auc": auc_a,
        "linear_scoring": {
            "day_penalties": {event_names_kr[k]: v for k, v in day_penalties_a.items()},
            "night_penalties": {event_names_kr[k]: v for k, v in night_penalties_a.items()}
        },
        "score_examples": {
            "risk_day": {"deduction": deduction_a_risk_day, "score": score_a_risk_day},
            "risk_night": {"deduction": deduction_a_risk_night, "score": score_a_risk_night},
            "safe_day": {"deduction": deduction_a_safe_day, "score": score_a_safe_day}
        }
    },
    "scenario_b": {
        "name": "3 Events (급가속, 급정거, 급회전, 과속 제외)",
        "n_features": len(feature_names_b),
        "feature_names": feature_names_b,
        "weights": weights_b,
        "bias": bias_b,
        "class_weight_positive": weight_pos_b,
        "metrics": metrics_b,
        "auc": auc_b,
        "linear_scoring": {
            "day_penalties": {event_names_kr[k]: v for k, v in day_penalties_b.items()},
            "night_penalties": {event_names_kr[k]: v for k, v in night_penalties_b.items()}
        },
        "score_examples": {
            "risk_day": {"deduction": deduction_b_risk_day, "score": score_b_risk_day},
            "risk_night": {"deduction": deduction_b_risk_night, "score": score_b_risk_night},
            "safe_day": {"deduction": deduction_b_safe_day, "score": score_b_safe_day}
        }
    }
}

output_file = "phase4f_phase_a_final_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"  [완료] 결과 저장: {output_file}")
print()

print("=" * 100)
print("[완료] Phase 4-F Phase A 최종 개선 완료!")
print("=" * 100)
print()

print("주요 결과:")
print(f"  Scenario A (4개): AUC {auc_a:.3f}, F1 {metrics_a['f1']:.3f}, Recall {metrics_a['recall']:.3f}")
print(f"  Scenario B (3개): AUC {auc_b:.3f}, F1 {metrics_b['f1']:.3f}, Recall {metrics_b['recall']:.3f}")
print()
print("다음 단계: python phase4f_phase_a_final_report.py")
print()
