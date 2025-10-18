"""
Phase 4G Step 3: 모델 학습 및 평가

모델:
1. Logistic Regression + Class Weight + Threshold 조정
2. Voting Ensemble (LR + RF + GBM)

시나리오:
A. 과속 고려 (Speeding included)
B. 과속 미고려 (Speeding excluded, 센서 기반만)
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io
import warnings
warnings.filterwarnings('ignore')

# UTF-8 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("Phase 4G Step 3: 모델 학습 및 평가")
print("=" * 80)

# ====================================================================================
# 1. 데이터 로드 및 전처리
# ====================================================================================
print("\n[1/7] 데이터 로드 중...")

with open('phase4g_combined_20k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

print(f"✅ 로드 완료: {len(df):,}개")

# 레이블
y = df['has_accident'].values

# 사고 비율
accident_rate = y.sum() / len(y) * 100
print(f"   사고 비율: {y.sum()}/{len(y)} ({accident_rate:.2f}%)")

# ====================================================================================
# 2. Scenario A: 과속 고려
# ====================================================================================
print("\n[2/7] Scenario A: 과속 고려 (모든 features 사용)")

# Features: 모든 이벤트 + 거리 + 야간
X_a = df[[
    'rapid_accel',
    'sudden_stop',
    'sharp_turn',
    'speeding',
    'distance_km'
]].copy()

# Is_Night 인코딩
X_a['is_night'] = (df['time_of_day'] == 'Night').astype(int)

print(f"   Features: {list(X_a.columns)}")
print(f"   Shape: {X_a.shape}")

# 스케일링
scaler_a = StandardScaler()
X_a_scaled = scaler_a.fit_transform(X_a)

# Train/Test Split (80/20)
X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(
    X_a_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {X_a_train.shape[0]:,} / Test: {X_a_test.shape[0]:,}")

# ====================================================================================
# 3. Scenario B: 과속 미고려 (센서 기반만)
# ====================================================================================
print("\n[3/7] Scenario B: 과속 미고려 (센서 기반 features만)")

# Features: 센서 기반 (가속도계) + 거리 + 야간
X_b = df[[
    'rapid_accel',
    'sudden_stop',
    'sharp_turn',
    'distance_km'
]].copy()

X_b['is_night'] = (df['time_of_day'] == 'Night').astype(int)

print(f"   Features: {list(X_b.columns)}")
print(f"   Shape: {X_b.shape}")

# 스케일링
scaler_b = StandardScaler()
X_b_scaled = scaler_b.fit_transform(X_b)

# Train/Test Split
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
    X_b_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {X_b_train.shape[0]:,} / Test: {X_b_test.shape[0]:,}")

# ====================================================================================
# 4. Model 1: Logistic Regression + Class Weight + Threshold 조정
# ====================================================================================
print("\n[4/7] Model 1: Logistic Regression + Class Weight + Threshold")

results = {}

for scenario_name, X_train, X_test, y_train, y_test in [
    ('Scenario A (과속 고려)', X_a_train, X_a_test, y_a_train, y_a_test),
    ('Scenario B (과속 미고려)', X_b_train, X_b_test, y_b_train, y_b_test)
]:
    print(f"\n   === {scenario_name} ===")

    # LR 모델 (class_weight='balanced')
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    lr.fit(X_train, y_train)

    # 확률 예측
    y_proba = lr.predict_proba(X_test)[:, 1]

    # Threshold 탐색 (0.3 ~ 0.7)
    best_threshold = 0.5
    best_recall = 0

    for threshold in np.arange(0.3, 0.71, 0.05):
        y_pred_temp = (y_proba >= threshold).astype(int)
        recall_temp = recall_score(y_test, y_pred_temp)

        if recall_temp > best_recall:
            best_recall = recall_temp
            best_threshold = threshold

    print(f"      최적 Threshold: {best_threshold:.2f} (Recall: {best_recall:.4f})")

    # 최적 threshold로 예측
    y_pred = (y_proba >= best_threshold).astype(int)

    # 평가
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"      Accuracy:  {acc:.4f}")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall:    {rec:.4f}")
    print(f"      F1-Score:  {f1:.4f}")
    print(f"      AUC-ROC:   {auc:.4f}")

    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    print(f"      Confusion Matrix:")
    print(f"         TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"         FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

    # 결과 저장
    results[f"LR_{scenario_name}"] = {
        'model': 'Logistic Regression',
        'scenario': scenario_name,
        'threshold': best_threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_proba': y_proba.tolist()
    }

# ====================================================================================
# 5. Model 2: Voting Ensemble (LR + RF + GBM)
# ====================================================================================
print("\n[5/7] Model 2: Voting Ensemble (LR + RF + GBM)")

for scenario_name, X_train, X_test, y_train, y_test in [
    ('Scenario A (과속 고려)', X_a_train, X_a_test, y_a_train, y_a_test),
    ('Scenario B (과속 미고려)', X_b_train, X_b_test, y_b_train, y_b_test)
]:
    print(f"\n   === {scenario_name} ===")

    # Base 모델들
    lr_base = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    rf_base = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    gbm_base = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    # Voting Ensemble (Soft voting)
    voting = VotingClassifier(
        estimators=[
            ('lr', lr_base),
            ('rf', rf_base),
            ('gbm', gbm_base)
        ],
        voting='soft'
    )

    print(f"      학습 중... (LR + RF + GBM)")
    voting.fit(X_train, y_train)

    # 확률 예측
    y_proba = voting.predict_proba(X_test)[:, 1]

    # Threshold 탐색
    best_threshold = 0.5
    best_recall = 0

    for threshold in np.arange(0.3, 0.71, 0.05):
        y_pred_temp = (y_proba >= threshold).astype(int)
        recall_temp = recall_score(y_test, y_pred_temp)

        if recall_temp > best_recall:
            best_recall = recall_temp
            best_threshold = threshold

    print(f"      최적 Threshold: {best_threshold:.2f} (Recall: {best_recall:.4f})")

    # 최적 threshold로 예측
    y_pred = (y_proba >= best_threshold).astype(int)

    # 평가
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"      Accuracy:  {acc:.4f}")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall:    {rec:.4f}")
    print(f"      F1-Score:  {f1:.4f}")
    print(f"      AUC-ROC:   {auc:.4f}")

    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    print(f"      Confusion Matrix:")
    print(f"         TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"         FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

    # Feature Importance (Random Forest)
    if scenario_name == 'Scenario A (과속 고려)':
        feature_names = ['급가속', '급정거', '급회전', '과속', '거리', '야간']
    else:
        feature_names = ['급가속', '급정거', '급회전', '거리', '야간']

    rf_model = voting.named_estimators_['rf']
    importances = rf_model.feature_importances_

    print(f"      Feature Importance (RF):")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"         {name}: {imp:.4f}")

    # 결과 저장
    results[f"Ensemble_{scenario_name}"] = {
        'model': 'Voting Ensemble',
        'scenario': scenario_name,
        'threshold': best_threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'feature_importance': {name: float(imp) for name, imp in zip(feature_names, importances)},
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_proba': y_proba.tolist()
    }

# ====================================================================================
# 6. 결과 비교
# ====================================================================================
print("\n[6/7] 결과 비교 분석")

print(f"\n{'모델':<20} {'시나리오':<25} {'Recall':<10} {'Precision':<10} {'F1':<10} {'AUC':<10}")
print("-" * 95)

for key, result in results.items():
    print(f"{result['model']:<20} {result['scenario']:<25} "
          f"{result['recall']:<10.4f} {result['precision']:<10.4f} "
          f"{result['f1']:<10.4f} {result['auc']:<10.4f}")

# 최고 성능 모델
best_recall_key = max(results.keys(), key=lambda k: results[k]['recall'])
best_f1_key = max(results.keys(), key=lambda k: results[k]['f1'])

print(f"\n   최고 Recall: {best_recall_key} ({results[best_recall_key]['recall']:.4f})")
print(f"   최고 F1-Score: {best_f1_key} ({results[best_f1_key]['f1']:.4f})")

# ====================================================================================
# 7. 결과 저장
# ====================================================================================
print("\n[7/7] 결과 저장 중...")

# JSON 저장
output_file = 'phase4g_model_results.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ 결과 저장 완료: {output_file}")

print("\n" + "=" * 80)
print("Phase 4G Step 3 완료!")
print("=" * 80)
print(f"\n다음 단계: python phase4g_step4_final_report.py")
