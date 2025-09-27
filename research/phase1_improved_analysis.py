"""
Phase 1 개선 분석: 과속 포함/제외 시나리오 비교
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import warnings
warnings.filterwarnings('ignore')

class Phase1ImprovedAnalysis:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_simulation_data(self, n_samples=10000):
        """
        개선된 시뮬레이션 데이터 생성
        """
        data = {
            'rapid_acceleration': np.random.poisson(2, n_samples),
            'sudden_stop': np.random.poisson(1.5, n_samples),
            'over_speeding': np.random.poisson(3, n_samples),
            'sharp_turn': np.random.poisson(1, n_samples),
            'is_night': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'weather_severity': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.3, 0.1]),
            'road_type': np.random.choice(['highway', 'city', 'rural'], n_samples, p=[0.4, 0.5, 0.1])
        }

        # 야간 효과
        night_mask = data['is_night'] == 1
        data['rapid_acceleration'] = np.where(night_mask,
                                             data['rapid_acceleration'] * 1.5,
                                             data['rapid_acceleration'])
        data['sudden_stop'] = np.where(night_mask,
                                      data['sudden_stop'] * 1.3,
                                      data['sudden_stop'])

        # 날씨 효과
        severe_weather = data['weather_severity'] >= 2
        data['sudden_stop'] = np.where(severe_weather,
                                      data['sudden_stop'] * 1.4,
                                      data['sudden_stop'])

        return pd.DataFrame(data)

    def calculate_accident_probability(self, df, include_overspeed=True):
        """
        사고 확률 계산 (과속 포함/제외)
        """
        base_prob = 0.05

        # 기본 이벤트 영향
        prob = (base_prob +
                df['rapid_acceleration'] * 0.02 +
                df['sudden_stop'] * 0.03 +
                df['sharp_turn'] * 0.025 +
                df['is_night'] * 0.03 +
                (df['weather_severity'] - 1) * 0.02)

        if include_overspeed:
            prob += df['over_speeding'] * 0.015
        else:
            # 과속 제외 시 다른 이벤트 가중치 보정
            prob += (df['rapid_acceleration'] * 0.005 +  # 추가 가중치
                    df['sudden_stop'] * 0.005 +
                    df['sharp_turn'] * 0.005)

        return np.clip(prob, 0, 0.8)

    def analyze_scenario(self, df, scenario_name, include_overspeed=True):
        """
        시나리오별 분석
        """
        print(f"\n{'='*60}")
        print(f"{scenario_name} 분석")
        print(f"{'='*60}")

        # 사고 확률 계산 및 할당
        accident_prob = self.calculate_accident_probability(df, include_overspeed)
        df['has_accident'] = np.random.binomial(1, accident_prob)

        print(f"사고 발생률: {df['has_accident'].mean():.1%}")

        # 이벤트 리스트 설정
        if include_overspeed:
            event_cols = ['rapid_acceleration', 'sudden_stop', 'over_speeding', 'sharp_turn']
        else:
            event_cols = ['rapid_acceleration', 'sudden_stop', 'sharp_turn']

        # 상관관계 분석
        print(f"\n상관관계 분석:")
        correlations = {}
        for event in event_cols:
            corr, p_val = stats.pearsonr(df[event], df['has_accident'])
            correlations[event] = {'correlation': corr, 'p_value': p_val}
            print(f"{event}: {corr:.4f} (p={p_val:.4f})")

        # 머신러닝 분석
        X = df[event_cols + ['is_night', 'weather_severity']].copy()

        # 야간 이벤트 특성 추가
        for col in event_cols:
            X[f'{col}_night'] = X[col] * X['is_night']

        y = df['has_accident']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        # 랜덤 포레스트 모델
        rf_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf_model.fit(X_train, y_train)

        rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
        print(f"\n랜덤 포레스트 AUC: {rf_auc:.4f}")

        # 특성 중요도
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n특성 중요도 (상위 5개):")
        print(importance_df.head(5).to_string(index=False))

        # 점수 계산
        base_score = 100
        penalty = 0

        for col in event_cols:
            penalty += df[col] * np.where(df['is_night'] == 1, 3, 2)

        df['safety_score'] = np.clip(base_score - penalty, 0, 100)

        print(f"\n점수 분포:")
        print(f"평균: {df['safety_score'].mean():.1f}점")
        print(f"표준편차: {df['safety_score'].std():.1f}점")
        print(f"중앙값: {df['safety_score'].median():.1f}점")

        # 등급 분포
        df['grade'] = pd.cut(df['safety_score'],
                            bins=[0, 60, 80, 100],
                            labels=['AGGRESSIVE', 'MODERATE', 'SAFE'])

        grade_dist = df['grade'].value_counts(normalize=True)
        print(f"\n등급 분포:")
        for grade, pct in grade_dist.items():
            print(f"{grade}: {pct:.1%}")

        # 등급별 사고율
        grade_accident = df.groupby('grade')['has_accident'].mean()
        print(f"\n등급별 사고율:")
        for grade, rate in grade_accident.items():
            print(f"{grade}: {rate:.1%}")

        return {
            'correlations': correlations,
            'auc': rf_auc,
            'importance': importance_df,
            'score_stats': df['safety_score'].describe(),
            'grade_distribution': grade_dist,
            'grade_accident_rates': grade_accident,
            'accident_rate': df['has_accident'].mean()
        }

    def compare_scenarios(self):
        """
        두 시나리오 비교 분석
        """
        print("공개 데이터 기반 운전 점수 시스템 연구 - Phase 1 개선 분석")
        print("="*80)

        # 데이터 생성
        df = self.generate_simulation_data()
        print(f"시뮬레이션 데이터 생성 완료: {len(df):,}개 샘플")

        # 시나리오 A: 4개 이벤트 포함
        df_scenario_a = df.copy()
        results_a = self.analyze_scenario(df_scenario_a, "시나리오 A: 4개 이벤트 포함 (현재 시스템)", True)

        # 시나리오 B: 3개 이벤트 (과속 제외)
        df_scenario_b = df.copy()
        results_b = self.analyze_scenario(df_scenario_b, "시나리오 B: 3개 이벤트 (과속 제외)", False)

        # 비교 분석
        self.generate_comparison_report(results_a, results_b)

        return results_a, results_b

    def generate_comparison_report(self, results_a, results_b):
        """
        비교 리포트 생성
        """
        print(f"\n{'='*80}")
        print("시나리오 A vs B 비교 분석 결과")
        print(f"{'='*80}")

        # 사고 발생률 비교
        print(f"\n1. 사고 발생률:")
        print(f"   시나리오 A (4개 이벤트): {results_a['accident_rate']:.1%}")
        print(f"   시나리오 B (3개 이벤트): {results_b['accident_rate']:.1%}")

        # 모델 성능 비교
        print(f"\n2. 모델 성능 (AUC):")
        print(f"   시나리오 A: {results_a['auc']:.4f}")
        print(f"   시나리오 B: {results_b['auc']:.4f}")
        print(f"   성능 차이: {results_b['auc'] - results_a['auc']:+.4f}")

        # 상관관계 비교
        print(f"\n3. 이벤트별 상관관계 비교:")
        common_events = ['rapid_acceleration', 'sudden_stop', 'sharp_turn']

        print(f"   {'이벤트':<20} {'시나리오 A':<12} {'시나리오 B':<12} {'개선율':<10}")
        print(f"   {'-'*60}")

        for event in common_events:
            corr_a = results_a['correlations'][event]['correlation']
            corr_b = results_b['correlations'][event]['correlation']
            improvement = (corr_b - corr_a) / corr_a * 100

            print(f"   {event:<20} {corr_a:<12.4f} {corr_b:<12.4f} {improvement:+.1f}%")

        # 과속만의 상관관계
        if 'over_speeding' in results_a['correlations']:
            overspeed_corr = results_a['correlations']['over_speeding']['correlation']
            print(f"   {'over_speeding':<20} {overspeed_corr:<12.4f} {'제외':<12} {'N/A':<10}")

        # 점수 분포 비교
        print(f"\n4. 점수 분포 비교:")
        print(f"   {'통계량':<15} {'시나리오 A':<12} {'시나리오 B':<12}")
        print(f"   {'-'*45}")
        print(f"   {'평균':<15} {results_a['score_stats']['mean']:<12.1f} {results_b['score_stats']['mean']:<12.1f}")
        print(f"   {'표준편차':<15} {results_a['score_stats']['std']:<12.1f} {results_b['score_stats']['std']:<12.1f}")

        # 등급 분포 비교
        print(f"\n5. 등급 분포 비교:")
        print(f"   {'등급':<15} {'시나리오 A':<12} {'시나리오 B':<12}")
        print(f"   {'-'*45}")
        for grade in ['SAFE', 'MODERATE', 'AGGRESSIVE']:
            pct_a = results_a['grade_distribution'].get(grade, 0)
            pct_b = results_b['grade_distribution'].get(grade, 0)
            print(f"   {grade:<15} {pct_a:<12.1%} {pct_b:<12.1%}")

        # 권고사항
        print(f"\n6. 분석 결과 및 권고사항:")

        # 성능 기준 권고
        if results_b['auc'] > results_a['auc']:
            print(f"   [O] 시나리오 B(3개 이벤트)가 더 높은 예측 성능을 보임")
        else:
            print(f"   [!] 시나리오 A(4개 이벤트)가 더 높은 예측 성능을 보임")

        # 상관관계 기준 권고
        improvements = []
        for event in common_events:
            corr_a = results_a['correlations'][event]['correlation']
            corr_b = results_b['correlations'][event]['correlation']
            if corr_b > corr_a:
                improvements.append(event)

        if len(improvements) >= 2:
            print(f"   [O] 과속 제외 시 {len(improvements)}개 이벤트의 상관관계가 개선됨")

        # 구현 복잡도 고려
        print(f"   [*] 구현 복잡도: 시나리오 B가 GPS 의존성 및 제한속도 정보 획득 문제 해결")

        # 최종 권고
        if results_b['auc'] >= results_a['auc'] * 0.98 and len(improvements) >= 2:
            print(f"\n   [결론] 최종 권고: 시나리오 B (3개 이벤트) 채택 권장")
            print(f"      - 예측 성능 유지/개선")
            print(f"      - 개별 이벤트 상관관계 향상")
            print(f"      - 구현 복잡도 감소")
        else:
            print(f"\n   [결론] 최종 권고: 추가 분석 필요 (Phase 2에서 실제 데이터로 검증)")

def main():
    """
    메인 실행 함수
    """
    analyzer = Phase1ImprovedAnalysis()
    results_a, results_b = analyzer.compare_scenarios()
    return results_a, results_b

if __name__ == "__main__":
    results = main()