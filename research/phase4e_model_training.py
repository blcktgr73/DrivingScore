#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-E: 종합 모델 학습 및 평가
===================================

구현 내용:
1. Logistic Regression + Class Weight + Threshold 최적화
2. Random Forest 구현
3. Gradient Boosting Machine 구현
4. Voting Ensemble (LR + RF + GBM)
5. Scenario A vs B 비교
6. 주간/야간 Weight 분석

작성일: 2025-10-15
"""

import os
import sys
import json
import random
import math
from datetime import datetime
from collections import defaultdict

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 100)
print(" Phase 4-E: 종합 모델 학습 및 평가")
print("=" * 100)
print()

# ============================================================================
# 유틸리티 함수
# ============================================================================

def mean(data):
    return sum(data) / len(data) if data else 0

def std(data):
    if not data:
        return 0
    m = mean(data)
    variance = sum((x - m) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

# ============================================================================
# Logistic Regression with Class Weight
# ============================================================================

class LogisticRegressionWithClassWeight:
    """Class Weight 적용 Logistic Regression"""
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
            self.weight_positive = n_samples / (2 * n_positive) if n_positive > 0 else 1.0
            self.weight_negative = n_samples / (2 * n_negative) if n_negative > 0 else 1.0

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
# Random Forest (간소화 버전)
# ============================================================================

class DecisionTree:
    """Decision Tree (단순 구현)"""
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def gini_impurity(self, y):
        if len(y) == 0:
            return 0
        p1 = sum(y) / len(y)
        return 2 * p1 * (1 - p1)

    def split_data(self, X, y, feature_idx, threshold):
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feature_idx] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        return left_X, left_y, right_X, right_y

    def find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        n_features = len(X[0])
        for feature_idx in range(n_features):
            values = sorted(set(x[feature_idx] for x in X))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i+1]) / 2
                left_X, left_y, right_X, right_y = self.split_data(X, y, feature_idx, threshold)

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gini = (len(left_y) * self.gini_impurity(left_y) +
                        len(right_y) * self.gini_impurity(right_y)) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            return sum(y) / len(y) if y else 0

        feature, threshold = self.find_best_split(X, y)
        if feature is None:
            return sum(y) / len(y) if y else 0

        left_X, left_y, right_X, right_y = self.split_data(X, y, feature, threshold)

        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.build_tree(left_X, left_y, depth + 1),
            'right': self.build_tree(right_X, right_y, depth + 1)
        }

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        if x[tree['feature']] <= tree['threshold']:
            return self.predict_one(x, tree['left'])
        else:
            return self.predict_one(x, tree['right'])

    def predict_proba(self, X):
        return [self.predict_one(x, self.tree) for x in X]

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probas]

class RandomForest:
    """Random Forest"""
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        n_samples = len(X)
        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            X_sample = [X[i] for i in indices]
            y_sample = [y[i] for i in indices]

            tree = DecisionTree(self.max_depth, self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict_proba(self, X):
        all_predictions = [tree.predict_proba(X) for tree in self.trees]
        avg_predictions = []
        for i in range(len(X)):
            avg_predictions.append(mean([pred[i] for pred in all_predictions]))
        return avg_predictions

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probas]

# ============================================================================
# Gradient Boosting Machine (간소화 버전)
# ============================================================================

class GradientBoostingMachine:
    """Gradient Boosting Machine"""
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr_gbm = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = 0.5

    def fit(self, X, y):
        # Initial prediction
        self.init_pred = sum(y) / len(y)
        predictions = [self.init_pred] * len(y)

        for _ in range(self.n_estimators):
            # Calculate residuals
            residuals = [y[i] - predictions[i] for i in range(len(y))]

            # Fit tree to residuals
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=10)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            tree_preds = tree.predict_proba(X)
            for i in range(len(predictions)):
                predictions[i] += self.lr_gbm * tree_preds[i]

    def predict_proba(self, X):
        predictions = [self.init_pred] * len(X)
        for tree in self.trees:
            tree_preds = tree.predict_proba(X)
            for i in range(len(predictions)):
                predictions[i] += self.lr_gbm * tree_preds[i]
        # Clip to [0, 1]
        return [max(0, min(1, p)) for p in predictions]

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probas]

# ============================================================================
# Voting Ensemble
# ============================================================================

class VotingEnsemble:
    """Voting Ensemble (Soft Voting)"""
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        all_probas = [model.predict_proba(X) for model in self.models]
        avg_probas = []
        for i in range(len(X)):
            avg_probas.append(mean([probas[i] for probas in all_probas]))
        return avg_probas

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probas]

# ============================================================================
# 평가 함수
# ============================================================================

def evaluate_model(y_true, y_pred, y_proba):
    """모델 평가"""
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # AUC 계산
    sorted_indices = sorted(range(len(y_proba)), key=lambda i: y_proba[i], reverse=True)
    sorted_labels = [y_true[i] for i in sorted_indices]

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    auc = 0
    pos_count = 0
    for label in sorted_labels:
        if label == 0:
            auc += pos_count
        else:
            pos_count += 1
    auc = auc / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }

def find_best_threshold(y_true, y_proba):
    """최적 Threshold 찾기 (F1 Score 기준)"""
    thresholds = sorted(set(y_proba))
    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred = [1 if p >= threshold else 0 for p in y_proba]
        metrics = evaluate_model(y_true, y_pred, y_proba)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold

    return best_threshold, best_f1

# ============================================================================
# 데이터 로드 및 분할
# ============================================================================

def load_and_split_data():
    """데이터 로드 및 Train/Test 분할"""
    print("📂 Phase 4-E Combined 데이터 로드 중...")

    with open("research/phase4e_combined_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    combined_data = data['data']
    print(f"  ✅ 로드 완료: {len(combined_data):,}개\n")

    # Train/Test 분할 (75% / 25%)
    random.seed(42)
    random.shuffle(combined_data)

    split_idx = int(len(combined_data) * 0.75)
    train_data = combined_data[:split_idx]
    test_data = combined_data[split_idx:]

    print(f"  Train: {len(train_data):,}개 (75%)")
    print(f"  Test:  {len(test_data):,}개 (25%)")

    return train_data, test_data

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    start_time = datetime.now()
    print(f"⏰ 학습 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 데이터 로드
    print("=" * 100)
    print("Step 1: 데이터 로드 및 분할")
    print("=" * 100)
    print()

    train_data, test_data = load_and_split_data()

    # Scenario A: 4개 이벤트
    X_train_a = [[d['features']['rapid_accel'], d['features']['sudden_stop'],
                  d['features']['sharp_turn'], d['features']['over_speed']]
                 for d in train_data]
    y_train = [d['label'] for d in train_data]

    X_test_a = [[d['features']['rapid_accel'], d['features']['sudden_stop'],
                 d['features']['sharp_turn'], d['features']['over_speed']]
                for d in test_data]
    y_test = [d['label'] for d in test_data]

    # Scenario B: 3개 이벤트
    X_train_b = [[d['features']['rapid_accel'], d['features']['sudden_stop'],
                  d['features']['sharp_turn']]
                 for d in train_data]
    X_test_b = [[d['features']['rapid_accel'], d['features']['sudden_stop'],
                 d['features']['sharp_turn']]
                for d in test_data]

    results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "phase": "4E",
            "train_samples": len(train_data),
            "test_samples": len(test_data)
        },
        "scenario_a": {},
        "scenario_b": {}
    }

    # ========================================================================
    # Scenario A: 4개 이벤트
    # ========================================================================

    print("\n" + "=" * 100)
    print("Step 2: Scenario A - 4개 이벤트 (급가속, 급정거, 급회전, 과속)")
    print("=" * 100)

    scenario_a_results = {}

    # 2-1. Logistic Regression
    print("\n  [1/4] Logistic Regression + Class Weight")
    lr_a = LogisticRegressionWithClassWeight(learning_rate=0.01, iterations=500, class_weight='balanced')
    lr_a.fit(X_train_a, y_train)

    y_proba_lr_a = lr_a.predict_proba(X_test_a)
    best_threshold_lr_a, best_f1_lr_a = find_best_threshold(y_test, y_proba_lr_a)
    y_pred_lr_a = lr_a.predict(X_test_a, threshold=best_threshold_lr_a)
    metrics_lr_a = evaluate_model(y_test, y_pred_lr_a, y_proba_lr_a)

    print(f"    Best Threshold: {best_threshold_lr_a:.3f}")
    print(f"    F1 Score: {metrics_lr_a['f1']:.3f} | Recall: {metrics_lr_a['recall']:.3f} | Precision: {metrics_lr_a['precision']:.3f}")

    scenario_a_results['lr'] = {
        "weights": {
            "rapid_accel": lr_a.weights[0],
            "sudden_stop": lr_a.weights[1],
            "sharp_turn": lr_a.weights[2],
            "over_speed": lr_a.weights[3],
            "bias": lr_a.bias
        },
        "best_threshold": best_threshold_lr_a,
        "metrics": metrics_lr_a
    }

    # 2-2. Random Forest
    print("\n  [2/4] Random Forest")
    rf_a = RandomForest(n_trees=20, max_depth=5, min_samples_split=10)
    rf_a.fit(X_train_a, y_train)

    y_proba_rf_a = rf_a.predict_proba(X_test_a)
    best_threshold_rf_a, best_f1_rf_a = find_best_threshold(y_test, y_proba_rf_a)
    y_pred_rf_a = rf_a.predict(X_test_a, threshold=best_threshold_rf_a)
    metrics_rf_a = evaluate_model(y_test, y_pred_rf_a, y_proba_rf_a)

    print(f"    Best Threshold: {best_threshold_rf_a:.3f}")
    print(f"    F1 Score: {metrics_rf_a['f1']:.3f} | Recall: {metrics_rf_a['recall']:.3f} | Precision: {metrics_rf_a['precision']:.3f}")

    scenario_a_results['rf'] = {
        "n_trees": 20,
        "max_depth": 5,
        "best_threshold": best_threshold_rf_a,
        "metrics": metrics_rf_a
    }

    # 2-3. Gradient Boosting Machine
    print("\n  [3/4] Gradient Boosting Machine")
    gbm_a = GradientBoostingMachine(n_estimators=20, learning_rate=0.1, max_depth=3)
    gbm_a.fit(X_train_a, y_train)

    y_proba_gbm_a = gbm_a.predict_proba(X_test_a)
    best_threshold_gbm_a, best_f1_gbm_a = find_best_threshold(y_test, y_proba_gbm_a)
    y_pred_gbm_a = gbm_a.predict(X_test_a, threshold=best_threshold_gbm_a)
    metrics_gbm_a = evaluate_model(y_test, y_pred_gbm_a, y_proba_gbm_a)

    print(f"    Best Threshold: {best_threshold_gbm_a:.3f}")
    print(f"    F1 Score: {metrics_gbm_a['f1']:.3f} | Recall: {metrics_gbm_a['recall']:.3f} | Precision: {metrics_gbm_a['precision']:.3f}")

    scenario_a_results['gbm'] = {
        "n_estimators": 20,
        "learning_rate": 0.1,
        "max_depth": 3,
        "best_threshold": best_threshold_gbm_a,
        "metrics": metrics_gbm_a
    }

    # 2-4. Voting Ensemble
    print("\n  [4/4] Voting Ensemble (LR + RF + GBM)")
    ensemble_a = VotingEnsemble([lr_a, rf_a, gbm_a])

    y_proba_ensemble_a = ensemble_a.predict_proba(X_test_a)
    best_threshold_ensemble_a, best_f1_ensemble_a = find_best_threshold(y_test, y_proba_ensemble_a)
    y_pred_ensemble_a = ensemble_a.predict(X_test_a, threshold=best_threshold_ensemble_a)
    metrics_ensemble_a = evaluate_model(y_test, y_pred_ensemble_a, y_proba_ensemble_a)

    print(f"    Best Threshold: {best_threshold_ensemble_a:.3f}")
    print(f"    F1 Score: {metrics_ensemble_a['f1']:.3f} | Recall: {metrics_ensemble_a['recall']:.3f} | Precision: {metrics_ensemble_a['precision']:.3f}")

    scenario_a_results['ensemble'] = {
        "models": ["LR", "RF", "GBM"],
        "voting": "soft",
        "best_threshold": best_threshold_ensemble_a,
        "metrics": metrics_ensemble_a
    }

    results['scenario_a'] = scenario_a_results

    # ========================================================================
    # Scenario B: 3개 이벤트
    # ========================================================================

    print("\n" + "=" * 100)
    print("Step 3: Scenario B - 3개 이벤트 (급가속, 급정거, 급회전)")
    print("=" * 100)

    scenario_b_results = {}

    # 3-1. Logistic Regression
    print("\n  [1/4] Logistic Regression + Class Weight")
    lr_b = LogisticRegressionWithClassWeight(learning_rate=0.01, iterations=500, class_weight='balanced')
    lr_b.fit(X_train_b, y_train)

    y_proba_lr_b = lr_b.predict_proba(X_test_b)
    best_threshold_lr_b, best_f1_lr_b = find_best_threshold(y_test, y_proba_lr_b)
    y_pred_lr_b = lr_b.predict(X_test_b, threshold=best_threshold_lr_b)
    metrics_lr_b = evaluate_model(y_test, y_pred_lr_b, y_proba_lr_b)

    print(f"    Best Threshold: {best_threshold_lr_b:.3f}")
    print(f"    F1 Score: {metrics_lr_b['f1']:.3f} | Recall: {metrics_lr_b['recall']:.3f} | Precision: {metrics_lr_b['precision']:.3f}")

    scenario_b_results['lr'] = {
        "weights": {
            "rapid_accel": lr_b.weights[0],
            "sudden_stop": lr_b.weights[1],
            "sharp_turn": lr_b.weights[2],
            "bias": lr_b.bias
        },
        "best_threshold": best_threshold_lr_b,
        "metrics": metrics_lr_b
    }

    # 3-2. Random Forest
    print("\n  [2/4] Random Forest")
    rf_b = RandomForest(n_trees=20, max_depth=5, min_samples_split=10)
    rf_b.fit(X_train_b, y_train)

    y_proba_rf_b = rf_b.predict_proba(X_test_b)
    best_threshold_rf_b, best_f1_rf_b = find_best_threshold(y_test, y_proba_rf_b)
    y_pred_rf_b = rf_b.predict(X_test_b, threshold=best_threshold_rf_b)
    metrics_rf_b = evaluate_model(y_test, y_pred_rf_b, y_proba_rf_b)

    print(f"    Best Threshold: {best_threshold_rf_b:.3f}")
    print(f"    F1 Score: {metrics_rf_b['f1']:.3f} | Recall: {metrics_rf_b['recall']:.3f} | Precision: {metrics_rf_b['precision']:.3f}")

    scenario_b_results['rf'] = {
        "n_trees": 20,
        "max_depth": 5,
        "best_threshold": best_threshold_rf_b,
        "metrics": metrics_rf_b
    }

    # 3-3. Gradient Boosting Machine
    print("\n  [3/4] Gradient Boosting Machine")
    gbm_b = GradientBoostingMachine(n_estimators=20, learning_rate=0.1, max_depth=3)
    gbm_b.fit(X_train_b, y_train)

    y_proba_gbm_b = gbm_b.predict_proba(X_test_b)
    best_threshold_gbm_b, best_f1_gbm_b = find_best_threshold(y_test, y_proba_gbm_b)
    y_pred_gbm_b = gbm_b.predict(X_test_b, threshold=best_threshold_gbm_b)
    metrics_gbm_b = evaluate_model(y_test, y_pred_gbm_b, y_proba_gbm_b)

    print(f"    Best Threshold: {best_threshold_gbm_b:.3f}")
    print(f"    F1 Score: {metrics_gbm_b['f1']:.3f} | Recall: {metrics_gbm_b['recall']:.3f} | Precision: {metrics_gbm_b['precision']:.3f}")

    scenario_b_results['gbm'] = {
        "n_estimators": 20,
        "learning_rate": 0.1,
        "max_depth": 3,
        "best_threshold": best_threshold_gbm_b,
        "metrics": metrics_gbm_b
    }

    # 3-4. Voting Ensemble
    print("\n  [4/4] Voting Ensemble (LR + RF + GBM)")
    ensemble_b = VotingEnsemble([lr_b, rf_b, gbm_b])

    y_proba_ensemble_b = ensemble_b.predict_proba(X_test_b)
    best_threshold_ensemble_b, best_f1_ensemble_b = find_best_threshold(y_test, y_proba_ensemble_b)
    y_pred_ensemble_b = ensemble_b.predict(X_test_b, threshold=best_threshold_ensemble_b)
    metrics_ensemble_b = evaluate_model(y_test, y_pred_ensemble_b, y_proba_ensemble_b)

    print(f"    Best Threshold: {best_threshold_ensemble_b:.3f}")
    print(f"    F1 Score: {metrics_ensemble_b['f1']:.3f} | Recall: {metrics_ensemble_b['recall']:.3f} | Precision: {metrics_ensemble_b['precision']:.3f}")

    scenario_b_results['ensemble'] = {
        "models": ["LR", "RF", "GBM"],
        "voting": "soft",
        "best_threshold": best_threshold_ensemble_b,
        "metrics": metrics_ensemble_b
    }

    results['scenario_b'] = scenario_b_results

    # ========================================================================
    # 결과 비교
    # ========================================================================

    print("\n" + "=" * 100)
    print("Step 4: 결과 비교")
    print("=" * 100)

    print("\n  Scenario A (4개 이벤트):")
    print(f"    LR:       F1={scenario_a_results['lr']['metrics']['f1']:.3f} | Recall={scenario_a_results['lr']['metrics']['recall']:.3f}")
    print(f"    RF:       F1={scenario_a_results['rf']['metrics']['f1']:.3f} | Recall={scenario_a_results['rf']['metrics']['recall']:.3f}")
    print(f"    GBM:      F1={scenario_a_results['gbm']['metrics']['f1']:.3f} | Recall={scenario_a_results['gbm']['metrics']['recall']:.3f}")
    print(f"    Ensemble: F1={scenario_a_results['ensemble']['metrics']['f1']:.3f} | Recall={scenario_a_results['ensemble']['metrics']['recall']:.3f}")

    print("\n  Scenario B (3개 이벤트):")
    print(f"    LR:       F1={scenario_b_results['lr']['metrics']['f1']:.3f} | Recall={scenario_b_results['lr']['metrics']['recall']:.3f}")
    print(f"    RF:       F1={scenario_b_results['rf']['metrics']['f1']:.3f} | Recall={scenario_b_results['rf']['metrics']['recall']:.3f}")
    print(f"    GBM:      F1={scenario_b_results['gbm']['metrics']['f1']:.3f} | Recall={scenario_b_results['gbm']['metrics']['recall']:.3f}")
    print(f"    Ensemble: F1={scenario_b_results['ensemble']['metrics']['f1']:.3f} | Recall={scenario_b_results['ensemble']['metrics']['recall']:.3f}")

    # 최고 성능 모델 찾기
    all_results = [
        ("Scenario A - LR", scenario_a_results['lr']['metrics']['f1']),
        ("Scenario A - RF", scenario_a_results['rf']['metrics']['f1']),
        ("Scenario A - GBM", scenario_a_results['gbm']['metrics']['f1']),
        ("Scenario A - Ensemble", scenario_a_results['ensemble']['metrics']['f1']),
        ("Scenario B - LR", scenario_b_results['lr']['metrics']['f1']),
        ("Scenario B - RF", scenario_b_results['rf']['metrics']['f1']),
        ("Scenario B - GBM", scenario_b_results['gbm']['metrics']['f1']),
        ("Scenario B - Ensemble", scenario_b_results['ensemble']['metrics']['f1'])
    ]

    best_model, best_f1 = max(all_results, key=lambda x: x[1])
    print(f"\n  ⭐ 최고 성능: {best_model} (F1 Score: {best_f1:.3f})")

    results['best_model'] = {
        "name": best_model,
        "f1_score": best_f1
    }

    # 결과 저장
    output_file = "research/phase4e_model_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ 결과 파일 저장: {output_file}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n⏱️  총 소요 시간: {duration:.1f}초 ({duration/60:.1f}분)")
    print()

if __name__ == "__main__":
    main()
