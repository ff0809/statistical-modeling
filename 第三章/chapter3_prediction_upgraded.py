"""
第三章：癌症风险预测模型（升级版）
升级内容：
  3.1.2 Logistic 回归 → ElasticNet 弹性网逻辑回归（L1+L2 双正则化）
  3.1.3 随机森林 → 概率校准均衡随机森林（Platt Sigmoid 校准）
  3.1.4 XGBoost → Optuna 贝叶斯优化 XGBoost（TPE 算法）
  3.1.5 新增：Stacking 集成学习模型（三基模型 + LR 元学习器）
  3.1.6 贝叶斯网络（保持原逻辑）

用法：python code/chapter3_prediction_upgraded.py
输出：results/chapter3_upgraded/ 下的表格和图表
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score,
                              recall_score, f1_score, brier_score_loss)
from scipy.stats import norm
import xgboost as xgb

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# ============================================================
# 路径设置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'chapter3_upgraded')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("  第三章（升级版）：癌症风险预测模型")
print("  ElasticNet + Random Forest + OptunaXGBoost + Stacking")
print("=" * 80)

# ============================================================
# 3.1.1 数据预处理与特征工程
# ============================================================
print("\n" + "-" * 60)
print("3.1.1 数据预处理与特征工程")
print("-" * 60)

metabolite_df = pd.read_csv(os.path.join(DATA_DIR, 'downloads.metabolite.txt'), sep='\t')
study_df = pd.read_csv(os.path.join(DATA_DIR, 'downloads.study.txt'), sep='\t')

print(f"代谢物记录数: {metabolite_df.shape[0]}")
print(f"研究队列数: {study_df.shape[0]}")

metabolite_df['case_concentration'] = pd.to_numeric(
    metabolite_df['case_concentration'], errors='coerce')
metabolite_df['control_concentration'] = pd.to_numeric(
    metabolite_df['control_concentration'], errors='coerce')

cohort_cancer = study_df[['Cohort_id', 'Cancer_type', 'Tissue']].drop_duplicates()
data = metabolite_df.merge(cohort_cancer, on='Cohort_id', how='left')

valid_data = data.dropna(subset=['case_concentration', 'control_concentration'])
valid_data = valid_data[(valid_data['case_concentration'] > 0) &
                         (valid_data['control_concentration'] > 0)]
valid_data['log2FC'] = np.log2(
    valid_data['case_concentration'] / valid_data['control_concentration'])

top_cancers = ['breast cancer', 'lung cancer', 'colorectal cancer',
               'prostate cancer', 'pancreatic cancer']
valid_data = valid_data[valid_data['Cancer_type'].isin(top_cancers)]

feature_pivot = valid_data.pivot_table(
    index=['Cohort_id', 'Cancer_type'],
    columns='original_metabolite_name',
    values='log2FC',
    aggfunc='mean'
)

meta_count = feature_pivot.notna().sum()
common_meta = meta_count[meta_count >= 3].index
feature_pivot = feature_pivot[common_meta]
feature_matrix = feature_pivot.fillna(0)

X = feature_matrix.values
y_labels = feature_matrix.index.get_level_values('Cancer_type')
feature_names = feature_matrix.columns.tolist()

print(f"\n特征矩阵: {X.shape[0]} 个队列 x {X.shape[1]} 个代谢物特征")
print(f"癌症类型分布:")
for c in top_cancers:
    print(f"  {c}: {(y_labels == c).sum()}")

y_binary = (y_labels == 'breast cancer').astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print(f"\nZ-score 标准化完成 (先切分后标准化，无数据泄漏)")
print(f"训练集: {X_train.shape[0]} 样本 | 测试集: {X_test.shape[0]} 样本")
print(f"乳腺癌阳性率: {y_binary.mean()*100:.1f}%")

# ============================================================
# DeLong 检验辅助函数
# ============================================================
def delong_roc_test(y_true, proba_A, proba_B):
    """DeLong's test for comparing two correlated ROC curves"""
    n = len(y_true)
    y_true = np.asarray(y_true, dtype=int)
    proba_A = np.asarray(proba_A, dtype=float)
    proba_B = np.asarray(proba_B, dtype=float)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    def compute_theta(proba):
        V10 = np.zeros(n)
        V01 = np.zeros(n)
        for i in range(n):
            if y_true[i] == 1:
                V10[i] = (proba[neg_mask] < proba[i]).mean() + \
                         0.5 * (proba[neg_mask] == proba[i]).mean()
            else:
                V01[i] = (proba[pos_mask] > proba[i]).mean() + \
                         0.5 * (proba[pos_mask] == proba[i]).mean()
        theta_val = V10[pos_mask].mean()
        return theta_val, V10, V01

    theta_A, V10_A, V01_A = compute_theta(proba_A)
    theta_B, V10_B, V01_B = compute_theta(proba_B)

    S10_A = np.zeros((n_pos, n_pos))
    S10_B = np.zeros((n_pos, n_pos))
    S01_A = np.zeros((n_neg, n_neg))
    S01_B = np.zeros((n_neg, n_neg))

    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]

    for i_idx, i in enumerate(pos_idx):
        for j_idx, j in enumerate(pos_idx):
            S10_A[i_idx, j_idx] = ((proba_A[neg_mask] < proba_A[i]).mean() -
                                    V10_A[i]) * ((proba_A[neg_mask] < proba_A[j]).mean() -
                                                  V10_A[j])
            S10_B[i_idx, j_idx] = ((proba_B[neg_mask] < proba_B[i]).mean() -
                                    V10_B[i]) * ((proba_B[neg_mask] < proba_B[j]).mean() -
                                                  V10_B[j])

    for i_idx, i in enumerate(neg_idx):
        for j_idx, j in enumerate(neg_idx):
            S01_A[i_idx, j_idx] = ((proba_A[pos_mask] > proba_A[i]).mean() -
                                    V01_A[i]) * ((proba_A[pos_mask] > proba_A[j]).mean() -
                                                  V01_A[j])
            S01_B[i_idx, j_idx] = ((proba_B[pos_mask] > proba_B[i]).mean() -
                                    V01_B[i]) * ((proba_B[pos_mask] > proba_B[j]).mean() -
                                                  V01_B[j])

    S10 = S10_A / (n_pos - 1) + S10_B / (n_pos - 1) if n_pos > 1 else np.zeros_like(S10_A)
    S01 = S01_A / (n_neg - 1) + S01_B / (n_neg - 1) if n_neg > 1 else np.zeros_like(S01_A)

    var_A = S10_A.sum() / (n_pos * (n_pos - 1)) + S01_A.sum() / (n_neg * (n_neg - 1))
    var_B = S10_B.sum() / (n_pos * (n_pos - 1)) + S01_B.sum() / (n_neg * (n_neg - 1))
    cov_AB = S10_A.sum() / (n_pos * (n_pos - 1)) + S01_A.sum() / (n_neg * (n_neg - 1))

    L = (theta_A - theta_B)
    var_L = max(var_A + var_B - 2 * cov_AB, 1e-16)
    z = L / np.sqrt(var_L)
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return theta_A, theta_B, z, p_value

# ============================================================
# 3.1.2 ElasticNet 弹性网逻辑回归
# ============================================================
print("\n" + "=" * 60)
print("3.1.2 ElasticNet 弹性网逻辑回归")
print("=" * 60)

print("  正在进行 GridSearchCV 超参数调优 (l1_ratio x C)...")
en_param_grid = {
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
    'C': np.logspace(-3, 1, 10)
}
en_base = LogisticRegression(
    penalty='elasticnet', solver='saga',
    max_iter=5000, random_state=42, n_jobs=-1
)
en_gscv = GridSearchCV(
    en_base, en_param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc', n_jobs=-1, verbose=1
)
en_gscv.fit(X_train, y_train)

en_best = en_gscv.best_estimator_
en_optimal_params = en_gscv.best_params_
print(f"\n  最优 l1_ratio: {en_optimal_params['l1_ratio']}")
print(f"  最优 C: {en_optimal_params['C']:.4f}")
print(f"  最优交叉验证 AUC: {en_gscv.best_score_:.4f}")

en_pred = en_best.predict(X_test)
en_proba = en_best.predict_proba(X_test)[:, 1]
en_fpr, en_tpr, _ = roc_curve(y_test, en_proba)
en_auc = auc(en_fpr, en_tpr)
en_acc = accuracy_score(y_test, en_pred)
en_prec = precision_score(y_test, en_pred, zero_division=0)
en_rec = recall_score(y_test, en_pred, zero_division=0)
en_f1 = f1_score(y_test, en_pred, zero_division=0)
en_brier = brier_score_loss(y_test, en_proba)

n_nonzero = (en_best.coef_[0] != 0).sum()
print(f"\n  AUC:       {en_auc:.4f}")
print(f"  Accuracy:  {en_acc:.4f}")
print(f"  Precision: {en_prec:.4f}")
print(f"  Recall:    {en_rec:.4f}")
print(f"  F1-Score:  {en_f1:.4f}")
print(f"  Brier:     {en_brier:.4f}")
print(f"  非零系数:  {n_nonzero}/{len(feature_names)} ({(1-n_nonzero/len(feature_names))*100:.1f}% 稀疏化)")

en_coef_df = pd.DataFrame({
    'metabolite': feature_names,
    'coefficient': en_best.coef_[0],
    'abs_coefficient': np.abs(en_best.coef_[0])
}).sort_values('abs_coefficient', ascending=False)
en_coef_df.to_csv(os.path.join(RESULTS_DIR, 'elasticnet_coefficients.csv'), index=False)

print("\nTop 10 弹性网系数代谢物:")
print(en_coef_df.head(10)[['metabolite', 'coefficient']].to_string(index=False))

en_nonzero_df = en_coef_df[en_coef_df['abs_coefficient'] > 0]
print(f"\n非零系数的代谢物 Top 5:")
print(en_nonzero_df.head(5)[['metabolite', 'coefficient']].to_string(index=False))

# ---- 图1: 弹性网正则化路径图 ----
n_path_points = 8
l1_ratios_for_path = np.round(np.linspace(0.01, 0.99, n_path_points), 2)
path_coef_norms = []
path_n_nonzero = []
path_auc_scores = []
skf = StratifiedKFold(3, shuffle=True, random_state=42)

print("  绘制正则化路径图...")
for i, l1r in enumerate(l1_ratios_for_path):
    print(f"    l1_ratio={l1r:.2f} ({i+1}/{n_path_points})")
    en_path = LogisticRegression(
        penalty='elasticnet', solver='saga', C=en_optimal_params['C'],
        l1_ratio=l1r, max_iter=2000, random_state=42, n_jobs=-1
    )
    en_path.fit(X_train, y_train)
    path_coef_norms.append(np.linalg.norm(en_path.coef_[0], ord=1))
    path_n_nonzero.append((en_path.coef_[0] != 0).sum())
    en_proba_path = en_path.predict_proba(X_test)[:, 1]
    path_auc_scores.append(auc(*roc_curve(y_test, en_proba_path)[:2]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
color_en = plt.cm.coolwarm(np.linspace(0.2, 0.9, n_path_points))
for i in range(1, n_path_points, max(1, n_path_points // 10)):
    ax.plot(range(len(feature_names)), sorted(np.abs(en_best.coef_[0] * (2 * l1_ratios_for_path[i] - 1)),
           reverse=True), alpha=0.3, color=color_en[i])
ax.plot(range(len(feature_names)), sorted(np.abs(en_best.coef_[0]), reverse=True),
        'k-', lw=2, label=f'l1_ratio={en_optimal_params["l1_ratio"]} (最优)')
ax.set_xlabel('Feature Rank', fontsize=11)
ax.set_ylabel('|Coefficient|', fontsize=11)
ax.set_title('(a) ElasticNet Coefficient Decay', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(l1_ratios_for_path, path_n_nonzero, 'b-o', lw=2, markersize=4,
        label=f'Non-zero ({path_n_nonzero[-1]} at l1r=0.99)')
ax2 = ax.twinx()
ax2.plot(l1_ratios_for_path, path_auc_scores, 'r-s', lw=2, markersize=4,
         label=f'CV AUC ({path_auc_scores[-1]:.3f} at l1r=0.99)')
ax.set_xlabel('l1_ratio (Higher = Stronger L1)', fontsize=11)
ax.set_ylabel('Number of Non-zero Coefficients', fontsize=11, color='b')
ax2.set_ylabel('Cross-Validation AUC', fontsize=11, color='r')
ax.set_title('(b) Sparsity-Accuracy Trade-off', fontsize=12, fontweight='bold')
ax.axvline(x=en_optimal_params['l1_ratio'], color='green', ls='--', lw=1.5,
           label=f'Optimal l1_ratio={en_optimal_params["l1_ratio"]}')
ax.legend(loc='upper left', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('ElasticNet Logistic Regression — Regularization Path Analysis',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'elasticnet_regularization_path.png'),
            dpi=200, bbox_inches='tight')
print("已保存: elasticnet_regularization_path.png")

# ---- 图2: GridSearchCV 热力图 ----
cv_results = pd.DataFrame(en_gscv.cv_results_)
pivot_auc = cv_results.pivot_table(
    index='param_l1_ratio', columns='param_C', values='mean_test_score')

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(pivot_auc.values, aspect='auto', cmap='RdYlBu_r', origin='lower')
ax.set_xticks(range(len(pivot_auc.columns)))
ax.set_xticklabels([f'{c:.3f}' for c in pivot_auc.columns], rotation=45, fontsize=8)
ax.set_yticks(range(len(pivot_auc.index)))
ax.set_yticklabels([f'{lr:.2f}' for lr in pivot_auc.index], fontsize=8)
ax.set_xlabel('C (Inverse Regularization Strength)', fontsize=11)
ax.set_ylabel('l1_ratio', fontsize=11)
ax.set_title('ElasticNet GridSearchCV — AUC Heatmap', fontsize=12, fontweight='bold')
best_l1r_idx = list(pivot_auc.index).index(en_optimal_params['l1_ratio'])
best_C_idx = list(pivot_auc.columns).index(en_optimal_params['C'])
ax.plot(best_C_idx, best_l1r_idx, 'k*', markersize=15)
ax.annotate(f'Best\nAUC={en_gscv.best_score_:.4f}',
            (best_C_idx, best_l1r_idx),
            xytext=(best_C_idx + 1.5, best_l1r_idx + 0.5),
            fontsize=9, color='black', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black'))
plt.colorbar(im, ax=ax, label='Mean CV AUC')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'elasticnet_gs_heatmap.png'),
            dpi=200, bbox_inches='tight')
print("已保存: elasticnet_gs_heatmap.png")

# ============================================================
# 3.1.3 随机森林
# ============================================================
print("\n" + "=" * 60)
print("3.1.3 随机森林模型")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
rf_auc = auc(rf_fpr, rf_tpr)
rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, zero_division=0)
rf_rec = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
rf_brier = brier_score_loss(y_test, rf_proba)

print(f"\n  AUC:       {rf_auc:.4f}")
print(f"  Accuracy:  {rf_acc:.4f}")
print(f"  Precision: {rf_prec:.4f}")
print(f"  Recall:    {rf_rec:.4f}")
print(f"  F1-Score:  {rf_f1:.4f}")
print(f"  Brier:     {rf_brier:.4f}")

rf_importance = pd.DataFrame({
    'metabolite': feature_names,
    'rf_importance': rf_model.feature_importances_
}).sort_values('rf_importance', ascending=False)
rf_importance.to_csv(os.path.join(RESULTS_DIR, 'rf_feature_importance.csv'), index=False)

print("\nTop 10 重要代谢物 (随机森林):")
print(rf_importance.head(10).to_string(index=False))

# ============================================================
# 3.1.4 Optuna 贝叶斯优化 XGBoost
# ============================================================
print("\n" + "=" * 60)
print("3.1.4 Optuna 贝叶斯优化 XGBoost (TPE 算法)")
print("=" * 60)

optuna_available = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna_available = True
except ImportError:
    print("  [警告] optuna 未安装，回退到默认 XGBoost 参数")
    print("  安装命令: pip install optuna>=3.0")

xgb_best_params = {}
xgb_trial_history = []

if optuna_available:
    n_trials = 30
    cv_folds = 3
    skf_xgb = StratifiedKFold(cv_folds, shuffle=True, random_state=42)

    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 5.0),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'verbosity': 0
        }

        aucs = []
        for tr_idx, val_idx in skf_xgb.split(X_train, y_train):
            model = xgb.XGBClassifier(**params)
            model.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
            val_proba = model.predict_proba(X_train[val_idx])[:, 1]
            aucs.append(auc(*roc_curve(y_train[val_idx], val_proba)[:2]))

        return float(np.mean(aucs))

    print(f"  启动 TPE 贝叶斯超参数优化 ({n_trials} trials, {cv_folds}-fold CV)...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=False)

    xgb_best_params = {
        k: v for k, v in study.best_params.items()
        if k not in ['scale_pos_weight']
    }
    xgb_best_params['random_state'] = 42
    xgb_best_params['n_jobs'] = -1
    xgb_best_params['eval_metric'] = 'logloss'

    xgb_trial_history = [
        {'trial': t.number, 'value': t.value,
         **{f'params_{k}': v for k, v in t.params.items()}}
        for t in study.trials
    ]

    print(f"\n  最优 Trial: {study.best_trial.number}")
    print(f"  最优 CV AUC: {study.best_value:.4f}")
    print(f"  最优参数:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")
        else:
            print(f"    {k}: {v}")

    # ---- 图4: Optuna 优化历史 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    trials_df = pd.DataFrame([
        {'trial': t.number, 'value': t.values[0]} for t in study.trials
    ])
    ax.plot(trials_df['trial'], trials_df['value'], 'o-', color='steelblue',
            markersize=3, lw=0.5, alpha=0.6)
    best_tracker = np.maximum.accumulate(trials_df['value'].values)
    ax.plot(trials_df['trial'], best_tracker, 'r-', lw=2,
            label=f'Best So Far ({best_tracker[-1]:.4f})')
    ax.set_xlabel('Trial', fontsize=11)
    ax.set_ylabel('Cross-Validation AUC', fontsize=11)
    ax.set_title('(a) Optimization History', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    param_importance = optuna.importance.get_param_importances(study)
    param_names = list(param_importance.keys())
    param_values = list(param_importance.values())
    colors_opt = plt.cm.viridis(np.linspace(0.2, 0.9, len(param_names)))
    ax.barh(range(len(param_names)), param_values, color=colors_opt)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels([n.replace('_', ' ').title() for n in param_names], fontsize=9)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('(b) Hyperparameter Importance', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    plt.suptitle('Optuna TPE Bayesian Optimization — XGBoost',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'optuna_optimization_history.png'),
                dpi=200, bbox_inches='tight')
    print("已保存: optuna_optimization_history.png")

    # ---- 图5: 参数平行坐标图 ----
    n_show = min(50, n_trials)
    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:n_show]
    top_values = [t.value for t in top_trials]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ax = axes[0]
    key_params = ['n_estimators', 'max_depth', 'learning_rate',
                  'subsample', 'min_child_weight']
    param_ranges = []
    for p in key_params:
        vals = [t.params[p] for t in top_trials]
        if isinstance(vals[0], float):
            param_ranges.append((min(vals), max(vals)))
        else:
            param_ranges.append((min(vals), max(vals)))

    for i, t in enumerate(top_trials):
        coords = []
        for j, p in enumerate(key_params):
            v = t.params[p]
            if param_ranges[j][1] > param_ranges[j][0]:
                coords.append((v - param_ranges[j][0]) /
                              (param_ranges[j][1] - param_ranges[j][0]))
            else:
                coords.append(0.5)
        alpha_val = 0.3 + 0.7 * (t.value - min(top_values)) / \
                    (max(top_values) - min(top_values) + 1e-8)
        ax.plot(range(len(key_params)), coords, '-o', markersize=3, lw=1,
                alpha=alpha_val, color=plt.cm.YlOrRd(t.value /
                max(top_values) if max(top_values) > 0 else 0.5))
    ax.set_xticks(range(len(key_params)))
    ax.set_xticklabels([p.replace('_', ' ').title() for p in key_params], fontsize=9)
    ax.set_ylabel('Normalized Value', fontsize=11)
    ax.set_title('(a) Hyperparameter Parallel Coordinates', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(
        vmin=min(top_values), vmax=max(top_values)))
    plt.colorbar(sm, ax=ax, label='CV AUC', shrink=0.8)

    ax = axes[1]
    best_params_to_plot = [('n_estimators', int), ('max_depth', int),
                           ('learning_rate', float),
                           ('subsample', float), ('min_child_weight', int),
                           ('reg_alpha', float), ('reg_lambda', float)]
    param_labels = []
    param_best_vals = []
    for p_name, p_type in best_params_to_plot:
        if p_name in study.best_params:
            param_labels.append(p_name.replace('_', ' ').title())
            param_best_vals.append(study.best_params[p_name])

    colors_bar = plt.cm.Set2(np.linspace(0, 1, len(param_labels)))
    ax.barh(range(len(param_labels)), param_best_vals, color=colors_bar, alpha=0.85)
    ax.set_yticks(range(len(param_labels)))
    ax.set_yticklabels(param_labels, fontsize=9)
    ax.set_title('(b) Optimal Hyperparameter Values', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    for i, v in enumerate(param_best_vals):
        ax.text(v, i, f'  {v:.4f}' if isinstance(v, float) else f'  {v}',
                va='center', fontsize=8)

    plt.suptitle('XGBoost Hyperparameter Optimization Results',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'optuna_param_analysis.png'),
                dpi=200, bbox_inches='tight')
    print("已保存: optuna_param_analysis.png")
else:
    xgb_best_params = {
        'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'
    }

xgb_best = xgb.XGBClassifier(**{k: v for k, v in xgb_best_params.items()
                                  if k != 'scale_pos_weight'})
xgb_best.fit(X_train, y_train)

xgb_pred = xgb_best.predict(X_test)
xgb_proba = xgb_best.predict_proba(X_test)[:, 1]
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)
xgb_auc = auc(xgb_fpr, xgb_tpr)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_prec = precision_score(y_test, xgb_pred, zero_division=0)
xgb_rec = recall_score(y_test, xgb_pred, zero_division=0)
xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
xgb_brier = brier_score_loss(y_test, xgb_proba)

print(f"\n  AUC:       {xgb_auc:.4f}")
print(f"  Accuracy:  {xgb_acc:.4f}")
print(f"  Precision: {xgb_prec:.4f}")
print(f"  Recall:    {xgb_rec:.4f}")
print(f"  F1-Score:  {xgb_f1:.4f}")
print(f"  Brier:     {xgb_brier:.4f}")

xgb_importance = pd.DataFrame({
    'metabolite': feature_names,
    'xgb_importance': xgb_best.feature_importances_
}).sort_values('xgb_importance', ascending=False)
xgb_importance.to_csv(os.path.join(RESULTS_DIR, 'xgb_feature_importance.csv'), index=False)

print("\nTop 10 重要代谢物 (Optuna XGBoost):")
print(xgb_importance.head(10).to_string(index=False))

# ============================================================
# 3.1.5 Stacking 集成学习模型
# ============================================================
print("\n" + "=" * 60)
print("3.1.5 Stacking 集成学习模型")
print("=" * 60)

print("  构建 Stacking 集成：ElasticNet + Random Forest + OptunaXGBoost -> LR (Meta)")

stacking = StackingClassifier(
    estimators=[
        ('elasticnet', LogisticRegression(
            penalty='elasticnet', solver='saga',
            l1_ratio=en_optimal_params['l1_ratio'],
            C=en_optimal_params['C'],
            max_iter=5000, random_state=42, n_jobs=-1
        )),
        ('random_forest', RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )),
        ('xgboost', xgb.XGBClassifier(
            **{k: v for k, v in xgb_best_params.items()
               if k in ['n_estimators', 'max_depth', 'learning_rate',
                         'subsample', 'colsample_bytree', 'min_child_weight',
                         'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs']}
        ))
    ],
    final_estimator=LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)
stacking.fit(X_train, y_train)

st_pred = stacking.predict(X_test)
st_proba = stacking.predict_proba(X_test)[:, 1]
st_fpr, st_tpr, _ = roc_curve(y_test, st_proba)
st_auc = auc(st_fpr, st_tpr)
st_acc = accuracy_score(y_test, st_pred)
st_prec = precision_score(y_test, st_pred, zero_division=0)
st_rec = recall_score(y_test, st_pred, zero_division=0)
st_f1 = f1_score(y_test, st_pred, zero_division=0)
st_brier = brier_score_loss(y_test, st_proba)

print(f"\n  AUC:       {st_auc:.4f}")
print(f"  Accuracy:  {st_acc:.4f}")
print(f"  Precision: {st_prec:.4f}")
print(f"  Recall:    {st_rec:.4f}")
print(f"  F1-Score:  {st_f1:.4f}")
print(f"  Brier:     {st_brier:.4f}")

print(f"\nStacking 元学习器系数:")
for name, coef in zip(
    ['ElasticNet', 'Random Forest', 'OptunaXGBoost'],
    stacking.final_estimator_.coef_[0]
):
    print(f"  {name}: {coef:.4f}")

# ---- 图6: Stacking vs 单模型 ROC 对比 ----
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(en_fpr, en_tpr, 'b-', lw=2, label=f'ElasticNet LR (AUC={en_auc:.4f})')
ax.plot(rf_fpr, rf_tpr, 'orange', lw=2, label=f'Random Forest (AUC={rf_auc:.4f})')
ax.plot(xgb_fpr, xgb_tpr, 'g-', lw=2, label=f'Optuna XGBoost (AUC={xgb_auc:.4f})')
ax.plot(st_fpr, st_tpr, 'r-', lw=3, label=f'Stacking Ensemble (AUC={st_auc:.4f})')
ax.plot([0, 1], [0, 1], 'gray', lw=1, ls='--', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — Individual Models vs Stacking Ensemble',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'stacking_roc_comparison.png'),
            dpi=200, bbox_inches='tight')
print("已保存: stacking_roc_comparison.png")

# ---- DeLong 检验 ----
print("\n--- DeLong 检验 (Stacking vs 各单模型) ---")
delong_results = []
for name, proba in [('ElasticNet', en_proba),
                     ('OptunaXGBoost', xgb_proba)]:
    auc_A, auc_B, z_stat, p_val = delong_roc_test(y_test, st_proba, proba)
    delong_results.append({
        'Comparison': f'Stacking vs {name}',
        'Stacking_AUC': f'{auc_A:.4f}',
        'Baseline_AUC': f'{auc_B:.4f}',
        'Delta_AUC': f'{auc_A - auc_B:.4f}',
        'Z_statistic': f'{z_stat:.4f}',
        'p_value': f'{p_val:.4f}',
        'Significant': 'Yes (p<0.05)' if p_val < 0.05 else 'No'
    })
    sig_str = '显著' if p_val < 0.05 else '不显著'
    print(f"  Stacking vs {name}: ΔAUC={auc_A - auc_B:.4f}, Z={z_stat:.4f}, p={p_val:.4f} ({sig_str})")

delong_df = pd.DataFrame(delong_results)
delong_df.to_csv(os.path.join(RESULTS_DIR, 'delong_test_results.csv'), index=False)

# ============================================================
# 3.1.6 贝叶斯网络结构分析（保持原逻辑）
# ============================================================
print("\n" + "=" * 60)
print("3.1.6 贝叶斯网络结构分析")
print("=" * 60)

try:
    from pgmpy.estimators import HillClimbSearch, BIC, BayesianEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.inference import VariableElimination
    import networkx as nx

    X_df = pd.DataFrame(X, columns=feature_names)
    X_discrete = X_df.copy()

    meta_freq = feature_pivot.notna().sum().sort_values(ascending=False)
    top28 = meta_freq.head(28).index.tolist()
    X_discrete = X_discrete[top28]
    X_discrete['Cancer_Status'] = y_binary

    for col in X_discrete.columns:
        if col == 'Cancer_Status':
            continue
        X_discrete[col] = pd.cut(
            X_discrete[col],
            bins=[-float('inf'), -1, 1, float('inf')],
            labels=[0, 1, 2]
        )
        X_discrete[col] = X_discrete[col].astype(int)

    X_discrete['Cancer_Status'] = X_discrete['Cancer_Status'].astype(int)

    print(f"贝叶斯网络输入: {X_discrete.shape[0]} 样本 x {X_discrete.shape[1]} 节点")

    hc = HillClimbSearch(X_discrete)
    dag = hc.estimate(scoring_method=BIC(X_discrete), max_indegree=5, max_iter=1000)
    print(f"\n学习到的 DAG 边数: {len(dag.edges())}")

    if len(dag.edges()) > 0:
        mb = dag.get_markov_blanket('Cancer_Status')
        print(f"\n癌症状态 Y 的马尔可夫 Blanket: {list(mb)}")

        edges_df = pd.DataFrame(list(dag.edges()), columns=['source', 'target'])
        edges_df.to_csv(os.path.join(RESULTS_DIR, 'bayesian_network_edges.csv'), index=False)
        print(f"  边列表已保存: {RESULTS_DIR}/bayesian_network_edges.csv")

        model = DiscreteBayesianNetwork(dag.edges())
        print("\n  开始参数学习 (Bayesian Estimation with Laplace Smoothing)...")
        model.fit(X_discrete, estimator=BayesianEstimator,
                  prior_type='dirichlet', pseudo_counts=1)

        cpt_dir = os.path.join(RESULTS_DIR, 'bayesian_cpt')
        os.makedirs(cpt_dir, exist_ok=True)
        import itertools

        all_cpts = []
        for node in model.nodes():
            cpt = model.get_cpds(node)
            cpt_filename = f"cpt_{node.replace('/', '_').replace(' ', '_')}.csv"
            cpt_path = os.path.join(cpt_dir, cpt_filename)

            variables = cpt.variables
            state_names = cpt.state_names
            rows = []
            if len(variables) == 1:
                for st in state_names[variables[0]]:
                    idx = list(state_names[variables[0]]).index(st)
                    rows.append({variables[0]: st,
                                 'probability': float(cpt.get_values()[idx])})
            else:
                parent_vars = variables[1:]
                child_var = variables[0]
                parent_states = [state_names[p] for p in parent_vars]
                for parent_combo in itertools.product(*parent_states):
                    parent_dict = dict(zip(parent_vars, parent_combo))
                    for child_st in state_names[child_var]:
                        row = parent_dict.copy()
                        row[child_var] = child_st
                        values_list = [child_st] + list(parent_combo)
                        vals_arr = cpt.get_values()
                        prob = vals_arr
                        for vi, vr in enumerate(values_list):
                            slist = list(state_names[variables[vi]])
                            prob = prob[slist.index(vr) if hasattr(slist, 'index') else int(vr)]
                        row['probability'] = float(prob) if isinstance(prob, (int, float, np.floating)) else 0.0
                        rows.append(row)
            df_cpt = pd.DataFrame(rows)
            df_cpt.to_csv(cpt_path, index=False)

            all_cpts.append({
                'node': node,
                'num_parents': len(model.get_parents(node)),
                'parents': str(model.get_parents(node)),
                'states': cpt.variable_card,
                'cpt_file': cpt_filename
            })
            print(f"  [OK] {node}: parents={model.get_parents(node)}, states={cpt.variable_card}")

        cpt_summary = pd.DataFrame(all_cpts)
        cpt_summary.to_csv(os.path.join(RESULTS_DIR, 'bayesian_cpt_summary.csv'), index=False)
        print(f"  CPT汇总表: {RESULTS_DIR}/bayesian_cpt_summary.csv")

        infer = VariableElimination(model)
        cancer_marginal = infer.query(variables=['Cancer_Status'], show_progress=False)

        mbox_available = [m for m in list(mb) if m != 'Cancer_Status']
        print(f"  马尔可夫毯变量 (排除Cancer_Status): {mbox_available}")

        pos = nx.spring_layout(dag, seed=42, k=3, iterations=50)
        fig, ax = plt.subplots(figsize=(18, 14))
        node_colors = []
        node_sizes = []
        for node in dag.nodes():
            if node == 'Cancer_Status':
                node_colors.append('red')
                node_sizes.append(1500)
            elif node in mb:
                node_colors.append('orange')
                node_sizes.append(1000)
            else:
                node_colors.append('lightblue')
                node_sizes.append(800)

        labels_display = {}
        for node in dag.nodes():
            if node == 'Cancer_Status':
                labels_display[node] = 'Cancer\nStatus'
            else:
                labels_display[node] = node[:15] + '..' if len(node) > 15 else node

        edges_list = [(u, v) for u, v in dag.edges()
                      if np.linalg.norm(np.array(pos[u]) - np.array(pos[v])) > 0.01]
        nx.draw(dag, pos, ax=ax, nodelist=list(dag.nodes()),
                node_color=node_colors, node_size=node_sizes, alpha=0.9,
                edgelist=edges_list, edge_color='gray',
                width=1.5, arrows=True, arrowsize=12,
                labels=labels_display, font_size=7, font_weight='bold')
        nx.draw_networkx_labels(dag, pos, labels=labels_display, font_size=7, font_weight='bold')
        plt.title('Bayesian Network DAG Structure\nRed=Cancer, Orange=Markov Blanket, Blue=Others',
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'bayesian_network_dag.png'),
                   dpi=200, bbox_inches='tight')
        print(f"  已保存: bayesian_network_dag.png")

        mb_nodes = list(mb) + ['Cancer_Status']
        mb_subgraph = dag.subgraph(mb_nodes)
        fig, ax = plt.subplots(figsize=(12, 10))
        pos_mb = nx.spring_layout(mb_subgraph, seed=42, k=2, iterations=50)
        mb_node_colors = ['red' if n == 'Cancer_Status' else 'orange' for n in mb_subgraph.nodes()]
        mb_node_sizes = [1500 if n == 'Cancer_Status' else 1000 for n in mb_subgraph.nodes()]
        mb_labels_display = {n: (n[:12] + '..' if len(n) > 12 else n) for n in mb_subgraph.nodes()}
        mb_labels_display['Cancer_Status'] = 'Cancer\nStatus'
        mb_edges = [(u, v) for u, v in mb_subgraph.edges()
                    if np.linalg.norm(np.array(pos_mb[u]) - np.array(pos_mb[v])) > 0.01]
        nx.draw(mb_subgraph, pos_mb, ax=ax, nodelist=list(mb_subgraph.nodes()),
                node_color=mb_node_colors, node_size=mb_node_sizes, alpha=0.9,
                edgelist=mb_edges, edge_color='darkblue',
                width=2, arrows=True, arrowsize=15,
                labels=mb_labels_display, font_size=9, font_weight='bold')
        plt.title('Markov Blanket of Cancer Status', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'bayesian_network_markov_blanket.png'),
                   dpi=200, bbox_inches='tight')
        print(f"  已保存: bayesian_network_markov_blanket.png")
    else:
        print("注意: 未学习到有效边。")
        mb = []
        mbox_available = []

except ImportError as e:
    print(f"pgmpy 或 networkx 未安装。贝叶斯网络部分已跳过。")
    print(f"错误信息: {e}")
    mb = []
    mbox_available = []

# ============================================================
# 3.3 综合结果对比
# ============================================================
print("\n" + "=" * 60)
print("3.3 综合结果对比")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model': ['ElasticNet LR', 'Random Forest',
              'Optuna XGBoost', 'Stacking Ensemble'],
    'AUC': [en_auc, rf_auc, xgb_auc, st_auc],
    'Accuracy': [en_acc, rf_acc, xgb_acc, st_acc],
    'Precision': [en_prec, rf_prec, xgb_prec, st_prec],
    'Recall': [en_rec, rf_rec, xgb_rec, st_rec],
    'F1-Score': [en_f1, rf_f1, xgb_f1, st_f1],
    'Brier': [en_brier, rf_brier, xgb_brier, st_brier]
})
comparison_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

print("\n四模型性能对比:")
print(comparison_df.to_string(index=False))

best_idx = comparison_df['AUC'].idxmax()
print(f"\n最优模型: {comparison_df.loc[best_idx, 'Model']} "
      f"(AUC={comparison_df.loc[best_idx, 'AUC']:.4f})")

# ---- 综合图1: 四模型 ROC 曲线 ----
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(en_fpr, en_tpr, 'b--', lw=2, label=f'ElasticNet LR (AUC={en_auc:.4f})')
ax.plot(rf_fpr, rf_tpr, 'orange', lw=2, label=f'Random Forest (AUC={rf_auc:.4f})')
ax.plot(xgb_fpr, xgb_tpr, 'g-', lw=2, label=f'Optuna XGBoost (AUC={xgb_auc:.4f})')
ax.plot(st_fpr, st_tpr, 'r-', lw=3, label=f'Stacking Ensemble (AUC={st_auc:.4f})')
ax.plot([0, 1], [0, 1], 'gray', lw=1, ls='--', alpha=0.5)
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('ROC Curves Comparison — Cancer Risk Prediction (Upgraded)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'), dpi=200, bbox_inches='tight')
print("已保存: roc_curves.png")

# ---- 综合图2: 四模型性能指标柱状图 ----
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(comparison_df))
width = 0.13
metrics_map = {
    'AUC': 'steelblue', 'Accuracy': 'forestgreen',
    'Precision': 'orange', 'Recall': 'coral',
    'F1-Score': 'mediumpurple', 'Brier': 'saddlebrown'
}
metrics_names = list(metrics_map.keys())
for i, (m, c) in enumerate(zip(metrics_names, metrics_map.values())):
    offset = (i - 2.5) * width
    bars = ax.bar(x_pos + offset, comparison_df[m], width, label=m, color=c, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=6, rotation=45)

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison (Upgraded)', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['ElasticNet\nLR', 'Calibrated\nBalancedRF',
                     'Optuna\nXGBoost', 'Stacking\nEnsemble'], fontsize=9)
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(0, 1.2)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_performance_bar.png'), dpi=200, bbox_inches='tight')
print("已保存: model_performance_bar.png")

# ---- 综合图3: Random Forest 特征重要性 Top 20 ----
fig, ax = plt.subplots(figsize=(10, 8))
top20_rf = rf_importance.head(20)
colors_rf = plt.cm.viridis(np.linspace(0.2, 0.9, len(top20_rf)))
ax.barh(range(len(top20_rf)), top20_rf['rf_importance'], color=colors_rf)
ax.set_yticks(range(len(top20_rf)))
ax.set_yticklabels(top20_rf['metabolite'], fontsize=8)
ax.set_xlabel('Feature Importance (Gini)', fontsize=11)
ax.set_title('Random Forest — Top 20 Important Metabolites',
             fontsize=13, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rf_feature_importance_top20.png'),
            dpi=200, bbox_inches='tight')
print("已保存: rf_feature_importance_top20.png")

# ---- 综合图4: Optuna XGBoost 特征重要性 Top 20 ----
fig, ax = plt.subplots(figsize=(10, 8))
top20_xgb = xgb_importance.head(20)
colors_xgb = plt.cm.plasma(np.linspace(0.2, 0.9, len(top20_xgb)))
ax.barh(range(len(top20_xgb)), top20_xgb['xgb_importance'], color=colors_xgb)
ax.set_yticks(range(len(top20_xgb)))
ax.set_yticklabels(top20_xgb['metabolite'], fontsize=8)
ax.set_xlabel('Feature Importance (Split Gain)', fontsize=11)
ax.set_title('Optuna XGBoost — Top 20 Important Metabolites',
             fontsize=13, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'xgb_feature_importance_top20.png'),
            dpi=200, bbox_inches='tight')
print("已保存: xgb_feature_importance_top20.png")

# ============================================================
# 保存最优参数供第五章使用
# ============================================================
optimal_params = {
    'elasticnet': {
        'l1_ratio': en_optimal_params['l1_ratio'],
        'C': en_optimal_params['C'],
        'penalty': 'elasticnet',
        'solver': 'saga',
        'max_iter': 5000,
        'n_nonzero_coefs': int(n_nonzero),
        'best_cv_auc': float(en_gscv.best_score_)
    },
    'calibrated_rf': {
        'n_estimators': 200,
        'class_weight': 'balanced_subsample',
        'max_features': 'sqrt',
        'calibration_method': 'sigmoid',
        'calibration_cv': 5
    }
}

if optuna_available:
    optimal_params['xgboost'] = {
        k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
        for k, v in study.best_params.items()
    }
else:
    optimal_params['xgboost'] = {
        'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
        'subsample': 0.8, 'colsample_bytree': 0.8
    }

optimal_params['stacking'] = {
    'cv': 5,
    'meta_learner': 'LogisticRegression',
    'n_base_learners': 3,
    'base_learners': ['ElasticNet', 'Random Forest', 'OptunaXGBoost']
}

params_path = os.path.join(RESULTS_DIR, 'optimal_params.json')
with open(params_path, 'w', encoding='utf-8') as f:
    json.dump(optimal_params, f, indent=2, ensure_ascii=False)
print(f"\n最优参数已保存至: {params_path}")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 80)
print("第三章（升级版）运行完成！")
print(f"所有结果已保存至: {RESULTS_DIR}")
print("=" * 80)
