# FULL COMPARISON SCRIPT 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Scikit-learn ensembles
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb

sns.set(style="whitegrid")

# ===================================
# Data
# ===================================
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# ===================================
# 1. AdaBoost (fixed & warning-free)
# ===================================
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=300,
    learning_rate=0.5,
    random_state=42
)
ada.fit(X_train, y_train)
results['AdaBoost'] = accuracy_score(y_test, ada.predict(X_test))

# ===================================
# 2. Gradient Boosting (sklearn)
# ===================================
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
results['GradientBoosting'] = accuracy_score(y_test, gb.predict(X_test))

# ===================================
# 3. XGBoost
# ===================================
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
results['XGBoost'] = accuracy_score(y_test, xgb_model.predict(X_test))

# ===================================
# 4. Bagging
# ===================================
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=300,
    max_samples=0.8,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train, y_train)
results['Bagging (Trees)'] = accuracy_score(y_test, bagging.predict(X_test))

# ===================================
# 5. Random Forest
# ===================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
results['RandomForest'] = accuracy_score(y_test, rf.predict(X_test))

# ===================================
# 6. Gaussian Naive Bayes
# ===================================
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
results['GaussianNB'] = accuracy_score(y_test, gnb.predict(X_test_scaled))

# ===================================
# Results
# ===================================
print("\n=== FINAL ACCURACY RANKING ===")
sorted_results = sorted(results.items(), key=lambda x: -x[1])
for i, (name, acc) in enumerate(sorted_results, 1):
    print(f"{i:2}. {name:20} → {acc:.4f}")

# ===================================
# Plot
# ===================================
plt.figure(figsize=(10, 6))
bars = plt.bar(
    [name for name, _ in sorted_results],
    [acc for _, acc in sorted_results],
    color=sns.color_palette("mako", len(results))
)
plt.title("Classification Methods Comparison\n(Breast Cancer Dataset)", fontsize=16)
plt.ylabel("Test Accuracy")
plt.ylim(0.90, 1.0)
plt.xticks(rotation=20, ha='right')

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h + 0.003,
             f'{h:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()