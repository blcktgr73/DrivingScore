"""
과속 제외 시나리오 분석
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def analyze_without_overspeed():
    """
    과속을 제외한 3개 이벤트만으로 분석
    """
    np.random.seed(42)
    n_samples = 10000

    # 과속을 제외한 3개 이벤트만 생성
    data = {
        'rapid_acceleration': np.random.poisson(2, n_samples),
        'sudden_stop': np.random.poisson(1.5, n_samples),
        'sharp_turn': np.random.poisson(1, n_samples),
        'is_night': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'weather_severity': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.3, 0.1]),
    }

    # 야간 증가 효과
    night_mask = data['is_night'] == 1
    data['rapid_acceleration'] = np.where(night_mask, data['rapid_acceleration'] * 1.5, data['rapid_acceleration'])
    data['sudden_stop'] = np.where(night_mask, data['sudden_stop'] * 1.3, data['sudden_stop'])

    # 과속 없이 사고 확률 계산
    accident_prob = (
        0.05 +
        data['rapid_acceleration'] * 0.025 +  # 가중치 증가
        data['sudden_stop'] * 0.035 +        # 가중치 증가
        data['sharp_turn'] * 0.030 +         # 가중치 증가
        data['is_night'] * 0.03 +
        (data['weather_severity'] - 1) * 0.02
    )
    accident_prob = np.clip(accident_prob, 0, 0.8)
    data['has_accident'] = np.random.binomial(1, accident_prob)

    df = pd.DataFrame(data)

    print("=== 과속 제외 시나리오 분석 ===")
    print(f"사고 발생률: {df['has_accident'].mean():.1%}")

    # 상관관계 분석
    event_cols = ['rapid_acceleration', 'sudden_stop', 'sharp_turn']
    print("\n상관관계 분석:")
    for event in event_cols:
        corr, p_val = stats.pearsonr(df[event], df['has_accident'])
        print(f"{event}: {corr:.4f} (p={p_val:.4f})")

    # 머신러닝 모델 성능
    X = df[event_cols + ['is_night', 'weather_severity']].copy()
    for col in event_cols:
        X[f'{col}_night'] = X[col] * X['is_night']

    y = df['has_accident']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습 및 평가
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    print(f"\n랜덤 포레스트 AUC (과속 제외): {rf_auc:.4f}")

    # 특성 중요도
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n특성 중요도 (과속 제외):")
    print(importance_df.head(6))

    # 현재 점수 계산 (과속 제외)
    base_score = 100
    df['penalty_without_overspeed'] = 0
    for col in event_cols:
        df['penalty_without_overspeed'] += df[col] * np.where(df['is_night'] == 1, 3, 2)

    df['score_without_overspeed'] = np.clip(base_score - df['penalty_without_overspeed'], 0, 100)

    print(f"\n점수 분포 (과속 제외):")
    print(f"평균: {df['score_without_overspeed'].mean():.1f}점")
    print(f"표준편차: {df['score_without_overspeed'].std():.1f}점")

    # 등급 분포
    df['grade_without_overspeed'] = pd.cut(df['score_without_overspeed'],
                                          bins=[0, 60, 80, 100],
                                          labels=['AGGRESSIVE', 'MODERATE', 'SAFE'])

    grade_dist = df['grade_without_overspeed'].value_counts(normalize=True)
    print(f"\n등급 분포 (과속 제외):")
    for grade, pct in grade_dist.items():
        print(f"{grade}: {pct:.1%}")

    return df, importance_df

def compare_with_overspeed():
    """
    과속 포함 vs 제외 비교
    """
    print("\n" + "="*50)
    print("과속 포함 vs 제외 비교 분석")
    print("="*50)

    # 과속 포함 (기존 분석 재현)
    np.random.seed(42)
    n_samples = 10000

    data_with = {
        'rapid_acceleration': np.random.poisson(2, n_samples),
        'sudden_stop': np.random.poisson(1.5, n_samples),
        'over_speeding': np.random.poisson(3, n_samples),
        'sharp_turn': np.random.poisson(1, n_samples),
        'is_night': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'weather_severity': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.3, 0.1]),
    }

    night_mask = data_with['is_night'] == 1
    data_with['rapid_acceleration'] = np.where(night_mask, data_with['rapid_acceleration'] * 1.5, data_with['rapid_acceleration'])
    data_with['sudden_stop'] = np.where(night_mask, data_with['sudden_stop'] * 1.3, data_with['sudden_stop'])

    accident_prob_with = (
        0.05 +
        data_with['rapid_acceleration'] * 0.02 +
        data_with['sudden_stop'] * 0.03 +
        data_with['over_speeding'] * 0.015 +
        data_with['sharp_turn'] * 0.025 +
        data_with['is_night'] * 0.03 +
        (data_with['weather_severity'] - 1) * 0.02
    )
    accident_prob_with = np.clip(accident_prob_with, 0, 0.8)
    data_with['has_accident'] = np.random.binomial(1, accident_prob_with)

    df_with = pd.DataFrame(data_with)

    # 비교 결과
    print(f"사고 발생률 - 과속 포함: {df_with['has_accident'].mean():.1%}")

    # 상관관계 비교
    events_with = ['rapid_acceleration', 'sudden_stop', 'over_speeding', 'sharp_turn']
    events_without = ['rapid_acceleration', 'sudden_stop', 'sharp_turn']

    print(f"\n상관관계 비교:")
    print(f"{'이벤트':<20} {'과속 포함':<12} {'과속 제외':<12}")
    print("-" * 50)

    df_without, _ = analyze_without_overspeed()

    for event in events_without:
        corr_with, _ = stats.pearsonr(df_with[event], df_with['has_accident'])
        corr_without, _ = stats.pearsonr(df_without[event], df_without['has_accident'])
        print(f"{event:<20} {corr_with:<12.4f} {corr_without:<12.4f}")

    # 과속만의 상관관계
    corr_overspeed, _ = stats.pearsonr(df_with['over_speeding'], df_with['has_accident'])
    print(f"{'over_speeding':<20} {corr_overspeed:<12.4f} {'N/A':<12}")

if __name__ == "__main__":
    df_without, importance_without = analyze_without_overspeed()
    compare_with_overspeed()