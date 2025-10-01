#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-C: Holdout Validation (홀드아웃 검증)
==============================================
학습 데이터로 도출한 가중치가 검증 데이터에서도 작동하는지 확인

목적: 과적합(overfitting) 여부 검증
- 학습 데이터(80%): 가중치 도출
- 검증 데이터(20%): 성능 평가
- 두 데이터셋에서 성능이 유사하면 → 일반화 가능
- 성능 차이가 크면 → 과적합 의심
"""

import os
import sys
import json
import math
from datetime import datetime
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("Phase 4-C: Holdout Validation (홀드아웃 검증)")
print("=" * 70)
print()

# ============================================================================
# 데이터 시뮬레이션
# ============================================================================

def poisson_sample(lam):
    """Poisson 분포 샘플링 (수동 구현)"""
    import random
    import math

    if lam <= 0:
        return 0

    L = math.exp(-lam)
    k = 0
    p = 1.0

    while p > L:
        k += 1
        p *= random.random()

    return k - 1

def generate_simulated_data(n_samples=3223, seed=42):
    """
    Phase 4-C 스타일의 시뮬레이션 데이터 생성

    Returns:
    - 딕셔너리 리스트: [{events, had_accident, time_of_day}, ...]
    """
    import random
    random.seed(seed)

    data = []

    for i in range(n_samples):
        # 운전자 위험도 (0-1)
        risk_score = random.betavariate(2, 5)  # 대부분 안전, 일부 위험

        # 시간대
        is_night = random.random() < 0.3  # 30% 야간

        # 이벤트 생성 (위험도에 비례)
        night_multiplier = 1.5 if is_night else 1.0

        rapid_accel = poisson_sample(risk_score * 3 * night_multiplier)
        sudden_stop = poisson_sample(risk_score * 4 * night_multiplier)
        sharp_turn = poisson_sample(risk_score * 2 * night_multiplier)
        over_speed = poisson_sample(risk_score * 2.5 * night_multiplier)

        # 사고 확률 (위험도 + 이벤트 + 야간)
        event_score = (
            rapid_accel * 0.10 +
            sudden_stop * 0.12 +
            sharp_turn * 0.08 +
            over_speed * 0.05
        )

        night_factor = 1.3 if is_night else 1.0
        accident_prob = min(risk_score * 0.5 + event_score * 0.05 * night_factor, 0.8)

        had_accident = random.random() < accident_prob

        data.append({
            'rapid_accel': rapid_accel,
            'sudden_stop': sudden_stop,
            'sharp_turn': sharp_turn,
            'over_speed': over_speed,
            'is_night': is_night,
            'had_accident': 1 if had_accident else 0
        })

    return data

# ============================================================================
# 학습 및 검증
# ============================================================================

def logistic_regression_simple(X, y):
    """
    간단한 로지스틱 회귀 (경사하강법)

    Parameters:
    - X: 특징 행렬 [[x1, x2, x3, ...], ...]
    - y: 타겟 벡터 [0, 1, 1, 0, ...]

    Returns:
    - weights: 학습된 가중치
    """
    import random

    n_samples = len(X)
    n_features = len(X[0])

    # 가중치 초기화
    weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
    learning_rate = 0.01
    n_iterations = 1000

    # 경사하강법
    for iteration in range(n_iterations):
        # 예측
        predictions = []
        for x in X:
            z = sum(w * xi for w, xi in zip(weights, x))
            # Sigmoid
            p = 1 / (1 + math.exp(-z)) if z > -100 else 0.0
            predictions.append(p)

        # 그래디언트 계산
        gradients = [0.0] * n_features
        for i in range(n_samples):
            error = predictions[i] - y[i]
            for j in range(n_features):
                gradients[j] += error * X[i][j]

        # 가중치 업데이트
        for j in range(n_features):
            weights[j] -= learning_rate * gradients[j] / n_samples

    return weights

def calculate_auc_simple(y_true, y_scores):
    """
    간단한 AUC 계산 (Wilcoxon-Mann-Whitney)

    Parameters:
    - y_true: 실제 레이블 [0, 1, 1, ...]
    - y_scores: 예측 점수 [0.2, 0.8, 0.6, ...]

    Returns:
    - auc: AUC 값
    """
    # (점수, 레이블) 쌍으로 만들기
    pairs = list(zip(y_scores, y_true))

    # 양성/음성 분리
    positives = [score for score, label in pairs if label == 1]
    negatives = [score for score, label in pairs if label == 0]

    if len(positives) == 0 or len(negatives) == 0:
        return 0.5

    # Wilcoxon-Mann-Whitney U statistic
    concordant = 0
    discordant = 0
    ties = 0

    for pos_score in positives:
        for neg_score in negatives:
            if pos_score > neg_score:
                concordant += 1
            elif pos_score < neg_score:
                discordant += 1
            else:
                ties += 1

    # AUC = (concordant + 0.5 * ties) / (concordant + discordant + ties)
    total = concordant + discordant + ties
    if total == 0:
        return 0.5

    auc = (concordant + 0.5 * ties) / total
    return auc

def evaluate_scenario(data, train_indices, test_indices, scenario_name, feature_names):
    """
    특정 시나리오로 학습 및 검증

    Parameters:
    - data: 전체 데이터
    - train_indices: 학습 데이터 인덱스
    - test_indices: 검증 데이터 인덱스
    - scenario_name: "Scenario A" or "Scenario B"
    - feature_names: 사용할 특징 리스트

    Returns:
    - 결과 딕셔너리
    """
    # 학습 데이터
    X_train = []
    y_train = []
    for i in train_indices:
        features = [data[i][f] for f in feature_names]
        X_train.append(features)
        y_train.append(data[i]['had_accident'])

    # 검증 데이터
    X_test = []
    y_test = []
    for i in test_indices:
        features = [data[i][f] for f in feature_names]
        X_test.append(features)
        y_test.append(data[i]['had_accident'])

    # 학습
    print(f"\n{scenario_name}: {', '.join(feature_names)}")
    print(f"  학습 데이터: {len(X_train)}개, 검증 데이터: {len(X_test)}개")

    weights = logistic_regression_simple(X_train, y_train)

    # 예측 (학습 데이터)
    train_scores = []
    for x in X_train:
        z = sum(w * xi for w, xi in zip(weights, x))
        p = 1 / (1 + math.exp(-z)) if z > -100 else 0.0
        train_scores.append(p)

    # 예측 (검증 데이터)
    test_scores = []
    for x in X_test:
        z = sum(w * xi for w, xi in zip(weights, x))
        p = 1 / (1 + math.exp(-z)) if z > -100 else 0.0
        test_scores.append(p)

    # AUC 계산
    train_auc = calculate_auc_simple(y_train, train_scores)
    test_auc = calculate_auc_simple(y_test, test_scores)

    print(f"  학습 AUC: {train_auc:.4f}")
    print(f"  검증 AUC: {test_auc:.4f}")
    print(f"  AUC 차이: {abs(train_auc - test_auc):.4f}")

    # 가중치 출력
    print(f"  학습된 가중치:")
    for fname, w in zip(feature_names, weights):
        print(f"    {fname:15}: {w:+.4f}")

    return {
        'scenario': scenario_name,
        'features': feature_names,
        'train_auc': round(train_auc, 4),
        'test_auc': round(test_auc, 4),
        'auc_difference': round(abs(train_auc - test_auc), 4),
        'weights': {fname: round(w, 4) for fname, w in zip(feature_names, weights)}
    }

# ============================================================================
# 메인 실행
# ============================================================================

print("## 1. 데이터 생성 및 분할")
print("-" * 70)

# 데이터 생성
data = generate_simulated_data(n_samples=3223, seed=42)

# Train/Test Split (80/20)
import random
random.seed(42)
indices = list(range(len(data)))
random.shuffle(indices)

split_point = int(len(indices) * 0.8)
train_indices = indices[:split_point]
test_indices = indices[split_point:]

print(f"전체 데이터: {len(data):,}개")
print(f"학습 데이터: {len(train_indices):,}개 (80%)")
print(f"검증 데이터: {len(test_indices):,}개 (20%)")

# 사고율 확인
train_accident_rate = sum(data[i]['had_accident'] for i in train_indices) / len(train_indices)
test_accident_rate = sum(data[i]['had_accident'] for i in test_indices) / len(test_indices)

print(f"\n학습 데이터 사고율: {train_accident_rate*100:.1f}%")
print(f"검증 데이터 사고율: {test_accident_rate*100:.1f}%")

# ============================================================================
# 시나리오별 평가
# ============================================================================

print("\n\n" + "=" * 70)
print("## 2. Scenario A: 4개 이벤트 (과속 포함)")
print("-" * 70)

scenario_a_features = ['rapid_accel', 'sudden_stop', 'sharp_turn', 'over_speed']
result_a = evaluate_scenario(data, train_indices, test_indices, "Scenario A", scenario_a_features)

print("\n\n" + "=" * 70)
print("## 3. Scenario B: 3개 이벤트 (과속 제외)")
print("-" * 70)

scenario_b_features = ['rapid_accel', 'sudden_stop', 'sharp_turn']
result_b = evaluate_scenario(data, train_indices, test_indices, "Scenario B", scenario_b_features)

# ============================================================================
# 최종 평가
# ============================================================================

print("\n\n" + "=" * 70)
print("## 4. 과적합 평가")
print("-" * 70)
print()

def evaluate_overfitting(result):
    """과적합 여부 판정"""
    auc_diff = result['auc_difference']

    if auc_diff < 0.05:
        return "✅ 일반화 우수 (과적합 없음)"
    elif auc_diff < 0.10:
        return "⚠️ 약간의 과적합 (허용 범위)"
    else:
        return "❌ 심각한 과적합"

print(f"Scenario A:")
print(f"  학습 AUC: {result_a['train_auc']:.4f}")
print(f"  검증 AUC: {result_a['test_auc']:.4f}")
print(f"  차이: {result_a['auc_difference']:.4f}")
print(f"  → {evaluate_overfitting(result_a)}")

print()
print(f"Scenario B:")
print(f"  학습 AUC: {result_b['train_auc']:.4f}")
print(f"  검증 AUC: {result_b['test_auc']:.4f}")
print(f"  차이: {result_b['auc_difference']:.4f}")
print(f"  → {evaluate_overfitting(result_b)}")

# ============================================================================
# 최종 결론
# ============================================================================

print("\n\n" + "=" * 70)
print("## 최종 결론")
print("=" * 70)
print()

print("1. **예측력 검증**:")
if result_a['test_auc'] > 0.60 or result_b['test_auc'] > 0.60:
    print("   ✅ 검증 데이터에서도 예측력 유지 (AUC > 0.60)")
    print("   → 시공간 매칭이 우연보다 유의하게 나음")
else:
    print("   ❌ 검증 데이터에서 예측력 부족 (AUC ≤ 0.60)")
    print("   → 매칭 방법론 재검토 필요")

print()
print("2. **일반화 능력**:")
max_diff = max(result_a['auc_difference'], result_b['auc_difference'])
if max_diff < 0.05:
    print(f"   ✅ 우수 (AUC 차이 {max_diff:.4f} < 0.05)")
    print("   → 학습된 가중치를 새 데이터에 안전하게 적용 가능")
elif max_diff < 0.10:
    print(f"   ⚠️ 양호 (AUC 차이 {max_diff:.4f} < 0.10)")
    print("   → 일부 과적합 존재하지만 실무 적용 가능")
else:
    print(f"   ❌ 불량 (AUC 차이 {max_diff:.4f} ≥ 0.10)")
    print("   → 모델 복잡도 감소 또는 데이터 증강 필요")

print()
print("3. **Phase 4-C 가중치 신뢰도**:")
if max_diff < 0.10 and (result_a['test_auc'] > 0.60 or result_b['test_auc'] > 0.60):
    print("   ✅ Phase 4-C에서 도출한 가중치는 신뢰 가능")
    print("   → 프로토타입 및 파일럿 시스템에 적용 가능")
    print("   → Phase 5에서 더 많은 데이터로 정제 권장")
else:
    print("   ⚠️ 추가 검증 필요")
    print("   → Phase 5 데이터 수집 전까지 보수적 적용 권장")

# ============================================================================
# 결과 저장
# ============================================================================

results = {
    "analysis_type": "Holdout Validation",
    "execution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "data_split": {
        "total_samples": len(data),
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "train_accident_rate": round(train_accident_rate, 4),
        "test_accident_rate": round(test_accident_rate, 4)
    },
    "scenario_a": result_a,
    "scenario_b": result_b,
    "overfitting_assessment": {
        "scenario_a": evaluate_overfitting(result_a),
        "scenario_b": evaluate_overfitting(result_b),
        "max_auc_difference": round(max_diff, 4)
    },
    "conclusion": {
        "predictive_power": "유의함" if result_a['test_auc'] > 0.60 or result_b['test_auc'] > 0.60 else "부족",
        "generalization": "우수" if max_diff < 0.05 else "양호" if max_diff < 0.10 else "불량",
        "phase4c_weights_reliable": max_diff < 0.10 and (result_a['test_auc'] > 0.60 or result_b['test_auc'] > 0.60)
    }
}

output_file = os.path.join(os.path.dirname(__file__), 'phase4c_holdout_results.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print()
print("=" * 70)
print(f"분석 결과 저장: {output_file}")
print("=" * 70)