# PERFECT & CLEAN FINAL VERSION (Nov 2025)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Scikit-learn
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# The big three
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

sns.set(style="whitegrid", font_scale=1.1)

# ===================================
# Data – IMPORTANT: use DataFrame so LightGBM/CatBoost love it
# ===================================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# ===================================
# 1. AdaBoost
# ===================================
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=500, learning_rate=0.5, random_state=42)
ada.fit(X_train, y_train)
results['AdaBoost'] = accuracy_score(y_test, ada.predict(X_test))

# ===================================
# 2. Sklearn GradientBoosting
# ===================================
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
results['GradientBoosting'] = accuracy_score(y_test, gb.predict(X_test))

# ===================================
# 3. XGBoost – slightly stronger params
# ===================================
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    reg_alpha=0,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train, y_train)
results['XGBoost'] = accuracy_score(y_test, xgb_model.predict(X_test))

# ===================================
# 4. LightGBM – proper training API (no warnings, early stopping)
# ===================================
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_test,  label=y_test, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1,
    'seed': 42
}

lgb_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_valid],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

lgb_pred = (lgb_model.predict(X_test) > 0.5).astype(int)
results['LightGBM'] = accuracy_score(y_test, lgb_pred)

# ===================================
# 5. CatBoost – default is already near-perfect
# ===================================
cat = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, verbose=False, random_seed=42)
cat.fit(X_train, y_train)
results['CatBoost'] = accuracy_score(y_test, cat.predict(X_test))

# ===================================
# 6. Bagging & Random Forest
# ===================================
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=500, max_samples=0.8, random_state=42, n_jobs=-1)
bagging.fit(X_train, y_train)
results['Bagging (Trees)'] = accuracy_score(y_test, bagging.predict(X_test))

rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results['RandomForest'] = accuracy_score(y_test, rf.predict(X_test))

# ===================================
# 7. Gaussian Naive Bayes
# ===================================
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
results['GaussianNB'] = accuracy_score(y_test, gnb.predict(X_test_scaled))

# ===================================
# FINAL RESULTS
# ===================================
sorted_results = sorted(results.items(), key=lambda x: -x[1])

print("\n" + "="*60)
print(" FINAL ACCURACY RANKING – Breast Cancer Dataset (Nov 2025)")
print("="*60)
for i, (name, acc) in enumerate(sorted_results, 1):
    print(f"{i:2}. {name:20} → {acc:.4f}  ({acc*100:.2f}%)")

# Plot
plt.figure(figsize=(11, 6))
bars = plt.bar([n for n, _ in sorted_results], [a for _, a in sorted_results],
               color=sns.color_palette("viridis", len(results)))
plt.title("Ultimate Classifier Showdown\nBreast Cancer Wisconsin Dataset", fontsize=16, pad=20)
plt.ylabel("Test Accuracy")
plt.ylim(0.92, 1.0)
plt.xticks(rotation=20, ha='right')
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h + 0.003, f'{h:.4f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()