"""
Phase 1: 기초 통계 분석 - 사고-이벤트 상관관계 및 환경적 위험 요인 분석 (시각화 제외)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    공개 데이터셋을 시뮬레이션하여 분석용 데이터 생성
    """
    np.random.seed(42)

    # 1. 시뮬레이션된 사고 데이터 (US Accidents Dataset 기반)
    n_samples = 10000

    # 기본 운전 이벤트 생성
    accidents_data = {
        'accident_severity': np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'rapid_acceleration': np.random.poisson(2, n_samples),
        'sudden_stop': np.random.poisson(1.5, n_samples),
        'over_speeding': np.random.poisson(3, n_samples),
        'sharp_turn': np.random.poisson(1, n_samples),
        'is_night': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'weather_severity': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.3, 0.1]),
        'road_type': np.random.choice(['highway', 'city', 'rural'], n_samples, p=[0.4, 0.5, 0.1]),
        'has_accident': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }

    # 야간 운전 시 위험 이벤트 증가 시뮬레이션
    night_mask = accidents_data['is_night'] == 1
    accidents_data['rapid_acceleration'] = np.where(night_mask,
                                                   accidents_data['rapid_acceleration'] * 1.5,
                                                   accidents_data['rapid_acceleration'])
    accidents_data['sudden_stop'] = np.where(night_mask,
                                             accidents_data['sudden_stop'] * 1.3,
                                             accidents_data['sudden_stop'])

    # 날씨에 따른 위험 이벤트 증가
    severe_weather_mask = accidents_data['weather_severity'] >= 2
    accidents_data['sudden_stop'] = np.where(severe_weather_mask,
                                             accidents_data['sudden_stop'] * 1.4,
                                             accidents_data['sudden_stop'])

    # 사고 확률을 이벤트 기반으로 조정
    accident_prob = (
        0.05 +
        accidents_data['rapid_acceleration'] * 0.02 +
        accidents_data['sudden_stop'] * 0.03 +
        accidents_data['over_speeding'] * 0.015 +
        accidents_data['sharp_turn'] * 0.025 +
        accidents_data['is_night'] * 0.03 +
        (accidents_data['weather_severity'] - 1) * 0.02
    )
    accident_prob = np.clip(accident_prob, 0, 0.8)
    accidents_data['has_accident'] = np.random.binomial(1, accident_prob)

    return pd.DataFrame(accidents_data)

def analyze_event_accident_correlation(df):
    """
    운전 이벤트와 사고 간의 상관관계 분석
    """
    print("=" * 60)
    print("1. 운전 이벤트-사고 상관관계 분석")
    print("=" * 60)

    # 운전 이벤트 컬럼들
    event_columns = ['rapid_acceleration', 'sudden_stop', 'over_speeding', 'sharp_turn']

    # 상관관계 계산
    correlations = {}
    for event in event_columns:
        # 피어슨 상관계수
        pearson_corr, pearson_p = stats.pearsonr(df[event], df['has_accident'])
        # 스피어만 상관계수 (순위 기반)
        spearman_corr, spearman_p = stats.spearmanr(df[event], df['has_accident'])

        correlations[event] = {
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p
        }

        print(f"\n{event}:")
        print(f"  Pearson 상관계수: {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"  Spearman 상관계수: {spearman_corr:.4f} (p={spearman_p:.4f})")

    return correlations

def analyze_environmental_risk_factors(df):
    """
    환경적 위험 요인 분석 (야간/주간, 날씨, 도로 유형)
    """
    print("\n" + "=" * 60)
    print("2. 환경적 위험 요인 분석")
    print("=" * 60)

    # 야간 vs 주간 사고율 분석
    night_accident_rate = df.groupby('is_night')['has_accident'].agg(['mean', 'count', 'std'])
    night_accident_rate.index = ['주간', '야간']
    print("\n야간/주간 사고율:")
    print(night_accident_rate)

    # 카이제곱 검정
    contingency_table = pd.crosstab(df['is_night'], df['has_accident'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\n야간 운전 영향 카이제곱 검정: χ²={chi2:.4f}, p={p_value:.4f}")

    # 날씨별 사고율 분석
    weather_accident_rate = df.groupby('weather_severity')['has_accident'].agg(['mean', 'count', 'std'])
    weather_accident_rate.index = ['맑음', '보통', '악천후']
    print("\n날씨별 사고율:")
    print(weather_accident_rate)

    # 도로 유형별 사고율 분석
    road_accident_rate = df.groupby('road_type')['has_accident'].agg(['mean', 'count', 'std'])
    print("\n도로 유형별 사고율:")
    print(road_accident_rate)

    # 야간/주간 이벤트 발생 빈도 비교
    event_cols = ['rapid_acceleration', 'sudden_stop', 'over_speeding', 'sharp_turn']
    day_events = df[df['is_night']==0][event_cols].mean()
    night_events = df[df['is_night']==1][event_cols].mean()

    print("\n야간/주간 운전 이벤트 평균 발생 횟수:")
    print("주간:", day_events.round(2).to_dict())
    print("야간:", night_events.round(2).to_dict())

    return {
        'night_vs_day': night_accident_rate,
        'weather_impact': weather_accident_rate,
        'road_type_impact': road_accident_rate,
        'night_chi2_test': {'chi2': chi2, 'p_value': p_value},
        'event_frequency': {'day': day_events, 'night': night_events}
    }

def calculate_optimal_weights(df):
    """
    머신러닝을 활용한 최적 가중치 계산
    """
    print("\n" + "=" * 60)
    print("3. 데이터 기반 최적 가중치 계산")
    print("=" * 60)

    # 특성 준비
    event_cols = ['rapid_acceleration', 'sudden_stop', 'over_speeding', 'sharp_turn']
    X = df[event_cols + ['is_night', 'weather_severity']].copy()
    y = df['has_accident']

    # 야간 이벤트 특성 추가 (현재 시스템 반영)
    for col in event_cols:
        X[f'{col}_night'] = X[col] * X['is_night']

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 로지스틱 회귀 모델
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)

    # 특성 중요도 (계수)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lr_model.coef_[0],
        'abs_coefficient': np.abs(lr_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)

    print("\n로지스틱 회귀 특성 중요도 (계수):")
    print(feature_importance.head(10))

    # 랜덤 포레스트 모델
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 특성 중요도
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n랜덤 포레스트 특성 중요도:")
    print(rf_importance.head(10))

    # 모델 성능 평가
    lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    print("\n모델 성능 비교:")
    print(f"로지스틱 회귀 AUC: {roc_auc_score(y_test, lr_pred_proba):.4f}")
    print(f"랜덤 포레스트 AUC: {roc_auc_score(y_test, rf_pred_proba):.4f}")

    # 현재 가중치와 비교
    current_weights = {
        'rapid_acceleration': 2,
        'sudden_stop': 2,
        'over_speeding': 2,
        'sharp_turn': 2,
        'night_multiplier': 1.5  # 3/2 = 1.5
    }

    # 새로운 가중치 제안 (특성 중요도 기반)
    base_weight = 2.0
    suggested_weights = {}

    print("\n제안된 가중치 vs 현재 가중치:")
    for col in event_cols:
        # 일반 이벤트 중요도
        day_importance = rf_importance[rf_importance['feature'] == col]['importance'].iloc[0]
        # 야간 이벤트 중요도
        night_col = f'{col}_night'
        night_importance = rf_importance[rf_importance['feature'] == night_col]['importance'].iloc[0]

        # 상대적 가중치 계산
        max_importance = rf_importance['importance'].max()
        day_weight = max(1.0, base_weight * (day_importance / max_importance * 2))
        night_weight = max(1.0, base_weight * (night_importance / max_importance * 3))

        suggested_weights[col] = {
            'day_weight': round(day_weight, 1),
            'night_weight': round(night_weight, 1),
            'current_day': current_weights[col],
            'current_night': current_weights[col] * current_weights['night_multiplier']
        }

        print(f"{col}:")
        print(f"  주간: {suggested_weights[col]['day_weight']} (현재: {suggested_weights[col]['current_day']})")
        print(f"  야간: {suggested_weights[col]['night_weight']} (현재: {suggested_weights[col]['current_night']})")

    return suggested_weights, feature_importance, rf_importance

def analyze_score_distribution(df):
    """
    운전자 점수 분포 분석 및 등급 분류 기준 최적화
    """
    print("\n" + "=" * 60)
    print("4. 점수 분포 및 등급 분류 기준 분석")
    print("=" * 60)

    # 현재 점수 계산 로직 시뮬레이션
    base_score = 100
    event_cols = ['rapid_acceleration', 'sudden_stop', 'over_speeding', 'sharp_turn']

    # 현재 시스템 점수 계산
    df['current_penalty'] = 0
    for col in event_cols:
        df['current_penalty'] += df[col] * np.where(df['is_night'] == 1, 3, 2)

    df['current_score'] = np.clip(base_score - df['current_penalty'], 0, 100)

    # 점수 분포 분석
    print(f"점수 기본 통계:")
    print(f"평균: {df['current_score'].mean():.2f}")
    print(f"표준편차: {df['current_score'].std():.2f}")
    print(f"중앙값: {df['current_score'].median():.2f}")
    print(f"최솟값: {df['current_score'].min():.2f}")
    print(f"최댓값: {df['current_score'].max():.2f}")

    # 분위수 분석
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    score_percentiles = np.percentile(df['current_score'], percentiles)

    print(f"\n점수 분위수:")
    for p, score in zip(percentiles, score_percentiles):
        print(f"{p}%: {score:.1f}점")

    # 현재 등급 분류
    df['current_grade'] = pd.cut(df['current_score'],
                                bins=[0, 60, 80, 100],
                                labels=['AGGRESSIVE', 'MODERATE', 'SAFE'])

    current_grade_dist = df['current_grade'].value_counts(normalize=True)
    print(f"\n현재 등급 분포:")
    for grade, pct in current_grade_dist.items():
        print(f"{grade}: {pct:.1%}")

    # ROC 기반 최적 임계값 탐색
    fpr, tpr, thresholds = roc_curve(df['has_accident'], 100 - df['current_score'])

    # Youden's J statistic으로 최적 임계값 찾기
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = 100 - thresholds[optimal_idx]

    print(f"\nROC 기반 최적 SAFE/AGGRESSIVE 구분점: {optimal_threshold:.1f}점")

    # 등급별 사고율 분석
    grade_accident_rates = df.groupby('current_grade')['has_accident'].agg(['mean', 'count'])
    print(f"\n등급별 사고율:")
    print(grade_accident_rates)

    return {
        'score_stats': df['current_score'].describe(),
        'grade_distribution': current_grade_dist,
        'optimal_threshold': optimal_threshold,
        'grade_accident_rates': grade_accident_rates
    }

def main():
    """
    메인 분석 실행 함수
    """
    print("공개 데이터 기반 운전 점수 시스템 연구 - Phase 1 기초 통계 분석")
    print("=" * 80)

    # 데이터 로드 및 준비
    df = load_and_prepare_data()
    print(f"데이터 로드 완료: {len(df):,}개 샘플")
    print(f"사고 발생률: {df['has_accident'].mean():.1%}")

    # 1. 사고-이벤트 상관관계 분석
    correlations = analyze_event_accident_correlation(df)

    # 2. 환경적 위험 요인 분석
    environmental_analysis = analyze_environmental_risk_factors(df)

    # 3. 최적 가중치 계산
    optimal_weights, lr_importance, rf_importance = calculate_optimal_weights(df)

    # 4. 점수 분포 및 등급 분류 기준 분석
    score_analysis = analyze_score_distribution(df)

    # 결과 요약
    print("\n" + "=" * 80)
    print("Phase 1 분석 결과 요약")
    print("=" * 80)

    print("\n1. 사고 예측력이 높은 이벤트 순위:")
    event_ranking = []
    for event, corr in correlations.items():
        event_ranking.append((event, abs(corr['pearson_corr'])))
    event_ranking.sort(key=lambda x: x[1], reverse=True)

    for i, (event, corr) in enumerate(event_ranking, 1):
        event_name = {
            'rapid_acceleration': '급가속',
            'sudden_stop': '급정거',
            'over_speeding': '과속',
            'sharp_turn': '급회전'
        }.get(event, event)
        print(f"  {i}. {event_name}: {corr:.4f}")

    print(f"\n2. 야간 운전 위험도: {environmental_analysis['night_vs_day'].loc['야간', 'mean']:.1%} vs {environmental_analysis['night_vs_day'].loc['주간', 'mean']:.1%}")
    print(f"3. 통계적 유의성: p={environmental_analysis['night_chi2_test']['p_value']:.4f}")
    print(f"4. 최적 등급 구분점: {score_analysis['optimal_threshold']:.1f}점")

    # 가중치 제안 요약
    print(f"\n5. 제안된 감점 가중치 (데이터 기반):")
    for event, weights in optimal_weights.items():
        event_name = {
            'rapid_acceleration': '급가속',
            'sudden_stop': '급정거',
            'over_speeding': '과속',
            'sharp_turn': '급회전'
        }.get(event, event)
        print(f"  {event_name}: 주간 -{weights['day_weight']}점, 야간 -{weights['night_weight']}점")

    return {
        'correlations': correlations,
        'environmental_analysis': environmental_analysis,
        'optimal_weights': optimal_weights,
        'score_analysis': score_analysis
    }

if __name__ == "__main__":
    # 분석 실행
    results = main()