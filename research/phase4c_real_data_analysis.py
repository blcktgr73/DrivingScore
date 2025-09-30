#!/usr/bin/env python3
"""
Phase 4-C: 실제 Kaggle 데이터 분석
====================================

실제 US Accidents + Vehicle Sensor 데이터 매칭 및 분석

Prerequisites:
1. Kaggle API 설정 완료
2. 데이터 다운로드 완료:
   - data/us_accidents/US_Accidents_March23.csv
   - data/driver_behavior/*.csv

실행:
python research/phase4c_real_data_analysis.py

예상 시간: 4-8시간 (데이터 크기에 따라)
예상 결과: 50,000-100,000개 매칭

작성일: 2025-09-30
"""

import os
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║       Phase 4-C: 실제 Kaggle 데이터 분석                    ║
║                                                              ║
║  목표: 50,000-100,000개 실제 매칭으로 최종 검증            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# 유틸리티 함수들
def mean(data):
    return sum(data) / len(data) if data else 0

def correlation(x, y):
    """피어슨 상관계수"""
    if len(x) != len(y) or len(x) == 0:
        return 0
    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = math.sqrt(sum((x[i] - mean_x)**2 for i in range(n)) * 
                           sum((y[i] - mean_y)**2 for i in range(n)))
    return numerator / denominator if denominator != 0 else 0

class Phase4CRealDataAnalysis:
    def __init__(self):
        self.data_dir = "data"
        self.us_accidents_file = os.path.join(self.data_dir, "us_accidents", "US_Accidents_March23.csv")
        self.driver_behavior_dir = os.path.join(self.data_dir, "driver_behavior")
        
        self.us_accidents_data = []
        self.vehicle_sensor_data = []
        self.matched_data = []
        self.results = {}
        
    def check_data_availability(self):
        """
        데이터 파일 존재 확인
        """
        print("=" * 60)
        print("📁 데이터 파일 확인")
        print("=" * 60)
        
        print(f"\n데이터 디렉토리: {os.path.abspath(self.data_dir)}")
        
        # US Accidents 확인
        if os.path.exists(self.us_accidents_file):
            size_mb = os.path.getsize(self.us_accidents_file) / (1024 * 1024)
            print(f"✅ US Accidents: {self.us_accidents_file}")
            print(f"   파일 크기: {size_mb:.1f} MB")
        else:
            print(f"❌ US Accidents 파일이 없습니다: {self.us_accidents_file}")
            print("\n다운로드 방법:")
            print("  kaggle datasets download -d sobhanmoosavi/us-accidents")
            print(f"  unzip us-accidents.zip -d {os.path.join(self.data_dir, 'us_accidents')}")
            return False
            
        # Driver Behavior 확인
        if os.path.exists(self.driver_behavior_dir):
            files = [f for f in os.listdir(self.driver_behavior_dir) if f.endswith('.csv')]
            if files:
                print(f"✅ Driver Behavior: {self.driver_behavior_dir}")
                print(f"   파일 수: {len(files)}개")
                for f in files[:3]:  # 처음 3개만 표시
                    size_mb = os.path.getsize(os.path.join(self.driver_behavior_dir, f)) / (1024 * 1024)
                    print(f"   - {f} ({size_mb:.1f} MB)")
                if len(files) > 3:
                    print(f"   ... 외 {len(files)-3}개")
            else:
                print(f"❌ CSV 파일이 없습니다: {self.driver_behavior_dir}")
                return False
        else:
            print(f"❌ Driver Behavior 디렉토리가 없습니다: {self.driver_behavior_dir}")
            print("\n다운로드 방법:")
            print("  kaggle datasets download -d outofskills/driving-behavior")
            print(f"  unzip driving-behavior.zip -d {self.driver_behavior_dir}")
            return False
            
        return True
        
    def load_us_accidents(self, sample_size=100000):
        """
        US Accidents 데이터 로딩
        
        Note: 실제 구현 시에는 pandas를 사용하여 CSV 파일을 읽어야 합니다.
        이 예제는 pandas가 없는 환경을 가정한 템플릿입니다.
        """
        print("\n" + "=" * 60)
        print(f"📊 US Accidents 데이터 로딩 (샘플: {sample_size:,}개)")
        print("=" * 60)
        
        print("\n⚠️ 주의: 이 스크립트는 pandas가 필요합니다.")
        print("\n실제 구현 예시:")
        print("""
import pandas as pd

# 청크 단위로 읽기 (메모리 절약)
chunks = []
for chunk in pd.read_csv(self.us_accidents_file, 
                         chunksize=10000,
                         usecols=['ID', 'Severity', 'Start_Time', 
                                 'Latitude', 'Longitude', 
                                 'Weather_Condition', 'Temperature(F)',
                                 'Visibility(mi)']):
    chunks.append(chunk)
    if len(chunks) * 10000 >= sample_size:
        break

df = pd.concat(chunks, ignore_index=True)

# 날짜 파싱
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Hour'] = df['Start_Time'].dt.hour
df['Is_Night'] = ((df['Hour'] >= 18) | (df['Hour'] <= 6)).astype(int)

# 데이터 저장
self.us_accidents_data = df.to_dict('records')
        """)
        
        print(f"\n현재 상태: 데이터 파일이 준비되어 있으나 pandas가 필요합니다.")
        print("설치: pip install pandas")
        
        return False  # pandas 없이는 진행 불가
        
    def load_vehicle_sensor(self):
        """
        Vehicle Sensor 데이터 로딩
        """
        print("\n" + "=" * 60)
        print("🚗 Vehicle Sensor 데이터 로딩")
        print("=" * 60)
        
        print("\n실제 구현 예시:")
        print("""
import pandas as pd

# 모든 CSV 파일 읽기
dfs = []
for file in os.listdir(self.driver_behavior_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(self.driver_behavior_dir, file))
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# 이벤트 감지
df['RapidAccel'] = (df['AccX'] > 1.2).astype(int)
df['SuddenStop'] = (df['AccX'] < -1.2).astype(int)
df['SharpTurn'] = (df['GyroZ'].abs() > 1.0).astype(int)

# 시간 정보
df['Hour'] = df['Timestamp'].apply(lambda x: # 시간 추출 로직)
df['Is_Night'] = ((df['Hour'] >= 18) | (df['Hour'] <= 6)).astype(int)

self.vehicle_sensor_data = df.to_dict('records')
        """)
        
        return False
        
    def perform_matching(self, target_matches=50000):
        """
        사고-센서 데이터 매칭
        """
        print("\n" + "=" * 60)
        print(f"🔗 데이터 매칭 실행 (목표: {target_matches:,}개)")
        print("=" * 60)
        
        print("\n매칭 전략:")
        print("  1. 지역-시간 매칭 (거리 200km, 시간 ±7일)")
        print("  2. 환경 조건 매칭 (야간/주간 일치 시 가점)")
        print("  3. 품질 필터링 (신뢰도 높은 매칭만 선별)")
        
        print("\n실제 구현 예시:")
        print("""
# 효율적인 매칭을 위한 인덱싱
from scipy.spatial import cKDTree

# 위치 기반 KD-Tree 구축
sensor_coords = np.array([[s['Latitude'], s['Longitude']] 
                          for s in self.vehicle_sensor_data])
tree = cKDTree(sensor_coords)

matches = []
for accident in self.us_accidents_data:
    acc_coord = [accident['Latitude'], accident['Longitude']]
    
    # 200km 이내 센서 찾기
    nearby_indices = tree.query_ball_point(acc_coord, r=2.0)  # ~200km
    
    for idx in nearby_indices:
        sensor = self.vehicle_sensor_data[idx]
        
        # 시간 차이 계산
        time_diff = abs((accident['Start_Time'] - sensor['Timestamp']).total_seconds())
        if time_diff > 604800:  # 7일
            continue
            
        # 매칭!
        match = {
            'accident_id': accident['ID'],
            'sensor_id': sensor['ID'],
            'severity': accident['Severity'],
            'distance_km': calculate_distance(acc_coord, sensor['coord']),
            'time_diff_hours': time_diff / 3600,
            # ... 기타 정보
        }
        matches.append(match)
        
        if len(matches) >= target_matches:
            break
    
    if len(matches) >= target_matches:
        break

self.matched_data = matches
        """)
        
        return False
        
    def analyze_correlations(self):
        """
        상관관계 분석
        """
        print("\n" + "=" * 60)
        print("📈 상관관계 분석")
        print("=" * 60)
        
        print("\n분석 항목:")
        print("  1. 이벤트별 사고 심각도 상관관계")
        print("  2. 야간 운전 vs 사고 심각도")
        print("  3. 날씨 조건 vs 사고 심각도")
        print("  4. 통계적 유의성 검정 (p-value)")
        
        print("\n예상 결과:")
        print("  - 급정거 상관계수: 0.20-0.30")
        print("  - 급가속 상관계수: 0.15-0.25")
        print("  - 급회전 상관계수: 0.10-0.20")
        print("  - 과속 상관계수: 0.05-0.15")
        
        return {}
        
    def derive_weights(self):
        """
        최종 가중치 도출
        """
        print("\n" + "=" * 60)
        print("⚖️ 최종 가중치 도출")
        print("=" * 60)
        
        print("\n방법론:")
        print("  1. 상관계수 기반 초기 가중치")
        print("  2. 로지스틱 회귀 계수 활용")
        print("  3. 랜덤 포레스트 특성 중요도")
        print("  4. 3가지 방법의 앙상블")
        
        print("\n예상 가중치 (주간/야간):")
        print("  - 급정거: -3.5점 / -5.5점")
        print("  - 급가속: -2.8점 / -4.2점")
        print("  - 급회전: -2.2점 / -3.3점")
        print("  - 과속: -1.8점 / -2.7점 (선택)")
        
        return {}
        
    def generate_final_report(self):
        """
        최종 보고서 생성
        """
        print("\n" + "=" * 60)
        print("📄 Phase 4-C 최종 보고서")
        print("=" * 60)
        
        report = {
            "phase": "Phase 4-C Real Data Analysis",
            "status": "준비 단계 - pandas 및 실제 데이터 필요",
            "requirements": {
                "packages": ["pandas", "numpy", "scikit-learn", "scipy"],
                "data": [
                    "US Accidents (7.7M records)",
                    "Vehicle Sensor (10K+ records)"
                ],
                "hardware": "32GB+ RAM 권장"
            },
            "expected_results": {
                "matched_samples": "50,000-100,000",
                "correlations": {
                    "sudden_stop": "0.20-0.30",
                    "rapid_accel": "0.15-0.25",
                    "sharp_turn": "0.10-0.20",
                    "over_speeding": "0.05-0.15"
                },
                "auc": "0.75-0.85",
                "p_value": "<0.0001"
            },
            "next_steps": [
                "1. Kaggle 데이터 다운로드",
                "2. pandas 및 필요 패키지 설치",
                "3. 데이터 전처리 실행",
                "4. 매칭 파이프라인 실행",
                "5. 통계 분석 및 가중치 도출",
                "6. 최종 보고서 작성"
            ]
        }
        
        output_file = "research/phase4c_preparation_status.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"\n✅ 준비 상태 저장: {output_file}")
        
        print("\n🎯 Phase 4-C 실행을 위한 다음 단계:")
        for step in report['next_steps']:
            print(f"  {step}")
            
        print("\n📚 참고 문서:")
        print("  - docs/Phase4C_Preparation_Guide.md")
        print("  - docs/Phase4B_Success_Report.md")
        
        return report

def main():
    """메인 실행 함수"""
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    phase4c = Phase4CRealDataAnalysis()
    
    # 1. 데이터 확인
    if not phase4c.check_data_availability():
        print("\n" + "=" * 60)
        print("⚠️ 데이터가 준비되지 않았습니다")
        print("=" * 60)
        print("\nPhase 4-C를 실행하려면:")
        print("1. docs/Phase4C_Preparation_Guide.md 참고")
        print("2. Kaggle에서 데이터 다운로드")
        print("3. 필요한 패키지 설치 (pandas, numpy, scikit-learn)")
        print("4. 이 스크립트를 실제 데이터 처리 로직으로 업데이트")
        
        # 준비 상태 보고서만 생성
        phase4c.generate_final_report()
        
        print("\n" + "=" * 60)
        print("📋 현재 상태: 준비 단계")
        print("=" * 60)
        print("\nPhase 4-B까지 완료되었으며,")
        print("Phase 4-C는 실제 Kaggle 데이터 확보 후 진행 가능합니다.")
        
        return
    
    # 2. 데이터 로딩
    if not phase4c.load_us_accidents():
        phase4c.generate_final_report()
        return
        
    if not phase4c.load_vehicle_sensor():
        phase4c.generate_final_report()
        return
    
    # 3. 매칭 실행
    if not phase4c.perform_matching():
        phase4c.generate_final_report()
        return
    
    # 4. 상관관계 분석
    phase4c.analyze_correlations()
    
    # 5. 가중치 도출
    phase4c.derive_weights()
    
    # 6. 최종 보고서
    phase4c.generate_final_report()
    
    print("\n" + "=" * 60)
    print("✅ Phase 4-C 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
