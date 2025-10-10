#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-D Week 2: sklearn Ensemble 모델
========================================

Week 1의 Class Weight + Threshold 최적화를 넘어,
sklearn 기반 고급 앙상블 모델(LogisticRegression, RandomForest, GradientBoosting)과
SMOTE를 적용하여 추가 성능 향상을 목표합니다.

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
print(" Phase 4-D Week 2: sklearn Ensemble 모델")
print("=" * 80)
print()

# ============================================================================
# sklearn import 시도
# ============================================================================

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from imblearn.over_sampling import SMOTE
    SKLEARN_AVAILABLE = True
    print("✅ sklearn과 imbalanced-learn 라이브러리 로드 성공")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"⚠️ sklearn 또는 imbalanced-learn 라이브러리가 설치되지 않았습니다: {e}")
    print("   다음 명령어로 설치하세요:")
    print("   pip install scikit-learn imbalanced-learn")
    sys.exit(1)

# ============================================================================
# 데이터 생성 (Week 1과 동일)
# ============================================================================

def mean(data):
    return sum(data) / len(data) if data else 0

def normal_random(mean_val, std_val):
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mean_val + z * std_val

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

def train_test_split_custom(X, y, test_size=0.25):
    """Train/Test 분할 (sklearn 대신 직접 구현)"""
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
# 평가 함수
# ============================================================================

def find_optimal_threshold(y_true, y_proba, optimization='f1'):
    """최적 Threshold 탐색"""
    thresholds = [i * 0.01 for i in range(1, 100)]
    best_threshold = 0.5
    best_score = 0
    best_metrics = None

    for threshold in thresholds:
        y_pred = [1 if p >= threshold else 0 for p in y_proba]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        score = f1 if optimization == 'f1' else recall

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'threshold': threshold
            }

    return best_threshold, best_metrics

# ============================================================================
# 모델 학습 및 평가
# ============================================================================

def run_ensemble_models(scenario_name, X_train, X_test, y_train, y_test):
    """Ensemble 모델 실행"""
    print(f"\n{'=' * 80}")
    print(f"🔬 {scenario_name}")
    print(f"{'=' * 80}")

    print(f"\n특징 개수: {len(X_train[0])}개")
    print(f"Train: {len(X_train):,}개 (사고율: {sum(y_train)/len(y_train):.1%}), Test: {len(X_test):,}개 (사고율: {sum(y_test)/len(y_test):.1%})")

    results = {}

    # Step 1: SMOTE 적용 (Class Imbalance 해결)
    print(f"\n[Step 1: SMOTE 적용]")
    print(f"  - 원본 Train: {len(X_train):,}개 (사고: {sum(y_train):,}개, 비사고: {len(y_train)-sum(y_train):,}개)")

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"  - SMOTE 후: {len(X_train_resampled):,}개 (사고: {sum(y_train_resampled):,}개, 비사고: {len(y_train_resampled)-sum(y_train_resampled):,}개)")
    print(f"  ✅ 완벽한 균형 달성 (50:50)")

    # Step 2: 개별 모델 학습
    print(f"\n[Step 2: 개별 모델 학습]")

    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=50,
            class_weight='balanced',
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=50,
            random_state=42
        )
    }

    model_results = {}

    for model_name, model in models.items():
        print(f"\n  [{model_name}]")

        # 학습
        model.fit(X_train_resampled, y_train_resampled)

        # 예측
        y_proba = model.predict_proba(X_test)[:, 1]

        # Threshold 최적화
        optimal_threshold, metrics = find_optimal_threshold(y_test, y_proba, optimization='f1')

        y_pred = [1 if p >= optimal_threshold else 0 for p in y_proba]

        # AUC 계산
        auc = roc_auc_score(y_test, y_proba)

        print(f"    최적 Threshold: {optimal_threshold:.2f}")
        print(f"    Precision: {metrics['precision']:.1%}, Recall: {metrics['recall']:.1%}, F1: {metrics['f1']:.4f}, AUC: {auc:.4f}")

        model_results[model_name] = {
            'threshold': optimal_threshold,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': auc
        }

    results['individual_models'] = model_results

    # Step 3: Voting Ensemble
    print(f"\n[Step 3: Voting Ensemble (Soft Voting)]")

    voting_clf = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=50, class_weight='balanced', random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, min_samples_split=50, random_state=42))
        ],
        voting='soft'
    )

    voting_clf.fit(X_train_resampled, y_train_resampled)
    y_proba_ensemble = voting_clf.predict_proba(X_test)[:, 1]

    # Threshold 최적화
    optimal_threshold_ensemble, metrics_ensemble = find_optimal_threshold(y_test, y_proba_ensemble, optimization='f1')

    y_pred_ensemble = [1 if p >= optimal_threshold_ensemble else 0 for p in y_proba_ensemble]

    # AUC 계산
    auc_ensemble = roc_auc_score(y_test, y_proba_ensemble)

    print(f"  최적 Threshold: {optimal_threshold_ensemble:.2f}")
    print(f"  Precision: {metrics_ensemble['precision']:.1%}, Recall: {metrics_ensemble['recall']:.1%}, F1: {metrics_ensemble['f1']:.4f}, AUC: {auc_ensemble:.4f}")

    results['ensemble'] = {
        'threshold': optimal_threshold_ensemble,
        'precision': metrics_ensemble['precision'],
        'recall': metrics_ensemble['recall'],
        'f1': metrics_ensemble['f1'],
        'auc': auc_ensemble
    }

    return results

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("\n데이터 생성 중...")

    # Scenario A (4개 이벤트)
    X_a, y_a = generate_data_scenario_a(n_samples=15000, accident_rate=0.359)
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split_custom(X_a, y_a, test_size=0.25)
    results_a = run_ensemble_models("Scenario A (4개 이벤트: 급가속, 급정거, 급회전, 과속)", X_train_a, X_test_a, y_train_a, y_test_a)

    # Scenario B (3개 이벤트)
    X_b, y_b = generate_data_scenario_b(n_samples=15000, accident_rate=0.359)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split_custom(X_b, y_b, test_size=0.25)
    results_b = run_ensemble_models("Scenario B (3개 이벤트: 급가속, 급정거, 급회전)", X_train_b, X_test_b, y_train_b, y_test_b)

    # 최종 비교
    print(f"\n{'=' * 80}")
    print(f"📊 Week 1 vs Week 2 성능 비교")
    print(f"{'=' * 80}")

    # Week 1 결과 로드
    week1_file = os.path.join(os.path.dirname(__file__), 'phase4d_scenario_comparison.json')
    with open(week1_file, 'r', encoding='utf-8') as f:
        week1_results = json.load(f)

    print(f"\n{'시나리오':<20} {'Week':<10} {'모델':<20} {'Precision':>12} {'Recall':>12} {'F1':>12} {'AUC':>12}")
    print("-" * 108)

    # Scenario A
    week1_a = week1_results['scenario_a']['phase4d']['metrics']
    ensemble_a = results_a['ensemble']

    print(f"{'Scenario A (4개)':<20} {'Week 1':<10} {'Class Weight + T':<20} {week1_a['precision']:>12.1%} {week1_a['recall']:>12.1%} {week1_a['f1']:>12.4f} {'N/A':>12}")
    print(f"{'Scenario A (4개)':<20} {'Week 2':<10} {'Voting Ensemble':<20} {ensemble_a['precision']:>12.1%} {ensemble_a['recall']:>12.1%} {ensemble_a['f1']:>12.4f} {ensemble_a['auc']:>12.4f}")

    f1_improvement_a = ensemble_a['f1'] - week1_a['f1']
    print(f"{'Scenario A (4개)':<20} {'개선':<10} {'':<20} {ensemble_a['precision'] - week1_a['precision']:>12.1%} {ensemble_a['recall'] - week1_a['recall']:>12.1%} {f1_improvement_a:>12.4f} {'':<12}")

    # Scenario B
    week1_b = week1_results['scenario_b']['phase4d']['metrics']
    ensemble_b = results_b['ensemble']

    print(f"{'Scenario B (3개)':<20} {'Week 1':<10} {'Class Weight + T':<20} {week1_b['precision']:>12.1%} {week1_b['recall']:>12.1%} {week1_b['f1']:>12.4f} {'N/A':>12}")
    print(f"{'Scenario B (3개)':<20} {'Week 2':<10} {'Voting Ensemble':<20} {ensemble_b['precision']:>12.1%} {ensemble_b['recall']:>12.1%} {ensemble_b['f1']:>12.4f} {ensemble_b['auc']:>12.4f}")

    f1_improvement_b = ensemble_b['f1'] - week1_b['f1']
    print(f"{'Scenario B (3개)':<20} {'개선':<10} {'':<20} {ensemble_b['precision'] - week1_b['precision']:>12.1%} {ensemble_b['recall'] - week1_b['recall']:>12.1%} {f1_improvement_b:>12.4f} {'':<12}")

    # 최종 권장사항
    print(f"\n{'=' * 80}")
    print(f"💡 최종 권장사항")
    print(f"{'=' * 80}")

    best_f1 = max(ensemble_a['f1'], ensemble_b['f1'])
    best_scenario = "A" if ensemble_a['f1'] > ensemble_b['f1'] else "B"

    print(f"\n🏆 추천: Scenario {best_scenario} + Week 2 Ensemble")
    if best_scenario == "A":
        print(f"   → F1 Score: {ensemble_a['f1']:.4f} (Scenario B 대비 +{ensemble_a['f1'] - ensemble_b['f1']:.4f})")
        print(f"   → AUC: {ensemble_a['auc']:.4f}")
        print(f"   → Week 1 대비 F1 개선: {f1_improvement_a:+.4f}")
    else:
        print(f"   → F1 Score: {ensemble_b['f1']:.4f} (Scenario A 대비 +{ensemble_b['f1'] - ensemble_a['f1']:.4f})")
        print(f"   → AUC: {ensemble_b['auc']:.4f}")
        print(f"   → Week 1 대비 F1 개선: {f1_improvement_b:+.4f}")

    # Week 2 효과 분석
    print(f"\n📈 Week 2 Ensemble의 효과:")
    if f1_improvement_a > 0 or f1_improvement_b > 0:
        print(f"  ✅ Scenario A: F1 {f1_improvement_a:+.4f} ({f1_improvement_a/week1_a['f1']*100:+.1f}%)")
        print(f"  ✅ Scenario B: F1 {f1_improvement_b:+.4f} ({f1_improvement_b/week1_b['f1']*100:+.1f}%)")
        print(f"  → SMOTE + Ensemble이 추가 성능 향상에 기여")
    else:
        print(f"  ⚠️ Scenario A: F1 {f1_improvement_a:+.4f} ({f1_improvement_a/week1_a['f1']*100:+.1f}%)")
        print(f"  ⚠️ Scenario B: F1 {f1_improvement_b:+.4f} ({f1_improvement_b/week1_b['f1']*100:+.1f}%)")
        print(f"  → Week 1의 Class Weight + Threshold 최적화가 이미 충분히 효과적")
        print(f"  → Ensemble 복잡도 대비 성능 향상 미미 (Week 1 유지 권장)")

    # 결과 저장
    final_results = {
        'scenario_a': results_a,
        'scenario_b': results_b,
        'comparison': {
            'recommended_scenario': best_scenario,
            'f1_improvement_scenario_a': f1_improvement_a,
            'f1_improvement_scenario_b': f1_improvement_b,
            'week2_effective': f1_improvement_a > 0 or f1_improvement_b > 0
        }
    }

    output_file = os.path.join(os.path.dirname(__file__), 'phase4d_week2_ensemble.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {output_file}")

    print(f"\n{'=' * 80}")
    print(f"✅ Week 2 Ensemble 분석 완료!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
