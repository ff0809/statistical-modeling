"""
第三章：癌症风险预测模型
对应写作框架：3.1 模型建立（LR/RF/XGBoost/贝叶斯网络）
              3.2 实验设计
              3.3 结果分析
              3.4 模型性能分析

用法：python code/chapter3_prediction.py
输出：results/chapter3/ 下的表格和图表
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import xgboost as xgb

# 设置全局中文字体（避免图表中文字符乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# ============================================================
# 路径设置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'chapter3')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("  第三章：癌症风险预测模型")
print("  基于代谢组学的癌症预测模型及其可解释性研究")
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

metabolite_df['case_concentration'] = pd.to_numeric(metabolite_df['case_concentration'], errors='coerce')
metabolite_df['control_concentration'] = pd.to_numeric(metabolite_df['control_concentration'], errors='coerce')

cohort_cancer = study_df[['Cohort_id', 'Cancer_type', 'Tissue']].drop_duplicates()
data = metabolite_df.merge(cohort_cancer, on='Cohort_id', how='left')

valid_data = data.dropna(subset=['case_concentration', 'control_concentration'])
valid_data = valid_data[(valid_data['case_concentration'] > 0) & (valid_data['control_concentration'] > 0)]

valid_data['log2FC'] = np.log2(valid_data['case_concentration'] / valid_data['control_concentration'])

top_cancers = ['breast cancer', 'lung cancer', 'colorectal cancer', 'prostate cancer', 'pancreatic cancer']
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

print(f"\n特征矩阵: {X.shape[0]} 个队列 × {X.shape[1]} 个代谢物特征")
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

print(f"\n训练集: {X_train.shape[0]} 样本 | 测试集: {X_test.shape[0]} 样本")
print(f"乳腺癌阳性率: {y_binary.mean()*100:.1f}%")

# ============================================================
# 3.1.2 Logistic 回归模型
# ============================================================
print("\n" + "=" * 60)
print("3.1.2 Logistic 回归模型")
print("=" * 60)

lr_model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
lr_auc = auc(lr_fpr, lr_tpr)
lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred, zero_division=0)
lr_rec = recall_score(y_test, lr_pred, zero_division=0)
lr_f1 = f1_score(y_test, lr_pred, zero_division=0)

print(f"  AUC:       {lr_auc:.4f}")
print(f"  Accuracy:  {lr_acc:.4f}")
print(f"  Precision: {lr_prec:.4f}")
print(f"  Recall:    {lr_rec:.4f}")
print(f"  F1-Score:  {lr_f1:.4f}")

feature_names = feature_matrix.columns.tolist()
lr_odds_ratio = pd.DataFrame({
    'metabolite': feature_names,
    'coefficient': lr_model.coef_[0],
    'odds_ratio': np.exp(lr_model.coef_[0])
}).sort_values('odds_ratio', ascending=False)
lr_odds_ratio.to_csv(os.path.join(RESULTS_DIR, 'lr_odds_ratio.csv'), index=False)

print("\nTop 10 优势比代谢物 (OR = e^w):")
print(lr_odds_ratio.head(10)[['metabolite', 'odds_ratio']].to_string(index=False))

# ============================================================
# 3.1.3 随机森林模型
# ============================================================
print("\n" + "=" * 60)
print("3.1.3 随机森林模型")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1, max_features='sqrt'
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

print(f"  AUC:       {rf_auc:.4f}")
print(f"  Accuracy:  {rf_acc:.4f}")
print(f"  Precision: {rf_prec:.4f}")
print(f"  Recall:    {rf_rec:.4f}")
print(f"  F1-Score:  {rf_f1:.4f}")

rf_importance = pd.DataFrame({
    'metabolite': feature_names,
    'rf_importance': rf_model.feature_importances_
}).sort_values('rf_importance', ascending=False)
rf_importance.to_csv(os.path.join(RESULTS_DIR, 'rf_feature_importance.csv'), index=False)

print("\nTop 10 重要代谢物 (随机森林 - 基尼重要性):")
print(rf_importance.head(10).to_string(index=False))

# ============================================================
# 3.1.4 XGBoost 梯度提升模型
# ============================================================
print("\n" + "=" * 60)
print("3.1.4 XGBoost 梯度提升模型")
print("=" * 60)

xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)
xgb_auc = auc(xgb_fpr, xgb_tpr)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_prec = precision_score(y_test, xgb_pred, zero_division=0)
xgb_rec = recall_score(y_test, xgb_pred, zero_division=0)
xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)

print(f"  AUC:       {xgb_auc:.4f}")
print(f"  Accuracy:  {xgb_acc:.4f}")
print(f"  Precision: {xgb_prec:.4f}")
print(f"  Recall:    {xgb_rec:.4f}")
print(f"  F1-Score:  {xgb_f1:.4f}")

xgb_importance = pd.DataFrame({
    'metabolite': feature_names,
    'xgb_importance': xgb_model.feature_importances_
}).sort_values('xgb_importance', ascending=False)
xgb_importance.to_csv(os.path.join(RESULTS_DIR, 'xgb_feature_importance.csv'), index=False)

print("\nTop 10 重要代谢物 (XGBoost - 分裂增益):")
print(xgb_importance.head(10).to_string(index=False))

# ============================================================
# 3.1.5 贝叶斯网络结构分析
# ============================================================
print("\n" + "=" * 60)
print("3.1.5 贝叶斯网络结构分析")
print("=" * 60)

try:
    from pgmpy.estimators import HillClimbSearch, BicScore
    from pgmpy.models import BayesianNetwork

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

    print(f"贝叶斯网络输入: {X_discrete.shape[0]} 样本 × {X_discrete.shape[1]} 节点")
    print(f"  - 代谢物节点: {len(top28)} 个")
    print(f"  - 癌症状态节点: 1 个")
    print(f"  - 边数上限: {X_discrete.shape[1] * (X_discrete.shape[1]-1) // 2}")

    hc = HillClimbSearch(X_discrete)
    dag = hc.estimate(scoring_method=BicScore(X_discrete), max_indegree=5, max_iter=1000)
    print(f"\n学习到的 DAG 边数: {len(dag.edges())}")

    if len(dag.edges()) > 0:
        mb = dag.get_markov_blanket('Cancer_Status')
        print(f"\n癌症状态 Y 的马尔可夫 Blanket: {list(mb)}")

        edges_df = pd.DataFrame(list(dag.edges()), columns=['source', 'target'])
        edges_df.to_csv(os.path.join(RESULTS_DIR, 'bayesian_network_edges.csv'), index=False)

        try:
            import networkx as nx
            G_nx = nx.DiGraph(dag.edges())
            edges_to_draw = [(u, v) for u, v in G_nx.edges() if u != v]
            G_nx_clean = nx.DiGraph()
            G_nx_clean.add_nodes_from(G_nx.nodes())
            G_nx_clean.add_edges_from(edges_to_draw)

            if len(G_nx_clean.edges()) == 0:
                print("注意: 没有有效边可绘制，跳过 DAG 图。")
            else:
                plt.figure(figsize=(24, 18))
                pos = nx.spring_layout(G_nx_clean, seed=42, k=5, iterations=200)

                def short_label(name, max_len=12):
                    return name if len(name) <= max_len else name[:max_len-3] + '...'

                labels = {node: short_label(node) for node in G_nx_clean.nodes()}

                nx.draw_networkx_nodes(G_nx_clean, pos, node_color='lightblue',
                                       node_size=2200, alpha=0.9)
                nx.draw_networkx_nodes(G_nx_clean, pos, nodelist=['Cancer_Status'],
                                       node_color='red', node_size=3500)
                nx.draw_networkx_edges(G_nx_clean, pos, arrows=True, arrowsize=35,
                                       edge_color='gray', width=2.5, alpha=0.7)
                nx.draw_networkx_labels(G_nx_clean, pos, labels, font_size=14,
                                        font_weight='bold')

                plt.title('贝叶斯网络 DAG 结构 (BIC + Hill-Climbing)', fontsize=18, fontweight='bold')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, 'bayesian_network_dag.png'),
                            dpi=200, bbox_inches='tight')
                print("已保存清晰版 DAG 图: bayesian_network_dag.png")
        except ImportError:
            print("networkx 未安装，跳过 DAG 可视化")
    else:
        print("注意: 未学习到有效边。样本量可能不足以支撑该网络结构。")

except ImportError:
    print("pgmpy 未安装。贝叶斯网络部分已跳过。")
    print("安装命令: pip install pgmpy")
    print("安装后重新运行即可。")

# ============================================================
# 3.3 结果分析
# ============================================================
print("\n" + "=" * 60)
print("3.3 结果分析 — 模型对比表")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'AUC': [lr_auc, rf_auc, xgb_auc],
    'Accuracy': [lr_acc, rf_acc, xgb_acc],
    'Precision': [lr_prec, rf_prec, xgb_prec],
    'Recall': [lr_rec, rf_rec, xgb_rec],
    'F1-Score': [lr_f1, rf_f1, xgb_f1]
})
comparison_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

print("\n三模型性能对比:")
print(comparison_df.to_string(index=False))

best_idx = comparison_df['AUC'].idxmax()
print(f"\n最优模型: {comparison_df.loc[best_idx, 'Model']} (AUC={comparison_df.loc[best_idx, 'AUC']:.4f})")

# ============================================================
# 3.4 模型性能分析 — 图表输出
# ============================================================
print("\n" + "-" * 60)
print("3.4 模型性能分析 — 生成图表")
print("-" * 60)

# ---- 图1: ROC 曲线对比 ----
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(lr_fpr, lr_tpr, 'b--', lw=2, label=f'Logistic Regression (AUC={lr_auc:.4f})')
ax.plot(rf_fpr, rf_tpr, 'orange', lw=2, label=f'Random Forest (AUC={rf_auc:.4f})')
ax.plot(xgb_fpr, xgb_tpr, 'g', lw=2, label=f'XGBoost (AUC={xgb_auc:.4f})')
ax.plot([0, 1], [0, 1], 'gray', lw=1, ls='--', alpha=0.5)
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('ROC Curves Comparison — Cancer Risk Prediction', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'), dpi=200, bbox_inches='tight')
print("已保存: roc_curves.png")

# ---- 图2: 三模型性能指标柱状图 ----
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(comparison_df))
width = 0.16
metrics_colors = ['steelblue', 'forestgreen', 'orange', 'coral', 'mediumpurple']
metrics_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
for i, (m, c) in enumerate(zip(metrics_names, metrics_colors)):
    offset = (i - 2) * width
    bars = ax.bar(x + offset, comparison_df[m], width, label=m, color=c, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'], fontsize=10)
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(0, 1.15)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_performance_bar.png'), dpi=200, bbox_inches='tight')
print("已保存: model_performance_bar.png")

# ---- 图3: RF 特征重要性 Top 20 ----
fig, ax = plt.subplots(figsize=(10, 8))
top20_rf = rf_importance.head(20)
colors_rf = plt.cm.viridis(np.linspace(0.2, 0.9, len(top20_rf)))
ax.barh(range(len(top20_rf)), top20_rf['rf_importance'], color=colors_rf)
ax.set_yticks(range(len(top20_rf)))
ax.set_yticklabels(top20_rf['metabolite'], fontsize=8)
ax.set_xlabel('Feature Importance (Gini Importance)', fontsize=11)
ax.set_title('Random Forest — Top 20 Important Metabolites', fontsize=13, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rf_feature_importance_top20.png'), dpi=200, bbox_inches='tight')
print("已保存: rf_feature_importance_top20.png")

# ---- 图4: XGBoost 特征重要性 Top 20 ----
fig, ax = plt.subplots(figsize=(10, 8))
top20_xgb = xgb_importance.head(20)
colors_xgb = plt.cm.plasma(np.linspace(0.2, 0.9, len(top20_xgb)))
ax.barh(range(len(top20_xgb)), top20_xgb['xgb_importance'], color=colors_xgb)
ax.set_yticks(range(len(top20_xgb)))
ax.set_yticklabels(top20_xgb['metabolite'], fontsize=8)
ax.set_xlabel('Feature Importance (Split Gain)', fontsize=11)
ax.set_title('XGBoost — Top 20 Important Metabolites', fontsize=13, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'xgb_feature_importance_top20.png'), dpi=200, bbox_inches='tight')
print("已保存: xgb_feature_importance_top20.png")

# ============================================================
# DeLong 检验 (附加)
# ============================================================
print("\n" + "-" * 60)
print("DeLong 检验 — 模型 AUC 差异显著性")
print("-" * 60)

# ============================================================
# DeLong 检验 (手动实现，避免 mlxtend 版本问题)
# ============================================================
print("\n" + "-" * 60)
print("DeLong 检验 — 模型 AUC 差异显著性")
print("-" * 60)

import numpy as np
from scipy.stats import norm

def delong_roc_variance(y_true, y_pred):
    """计算单个模型 AUC 的 DeLong 方差"""
    n = len(y_true)
    n1 = np.sum(y_true)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return 0.0, 0.0
    
    # 计算结构分量 V10 和 V01
    # 根据 DeLong (1988) 论文公式
    y_pred_pos = y_pred[y_true == 1]
    y_pred_neg = y_pred[y_true == 0]
    
    # 核函数 psi(X,Y) = I(X<Y) + 0.5*I(X=Y)
    def psi(a, b):
        return np.where(a < b, 1.0, np.where(a == b, 0.5, 0.0))
    
    # 计算 V10 = E[psi(X, Y) | Y=1]
    v10 = np.zeros(n1)
    for i in range(n1):
        v10[i] = np.mean(psi(y_pred_neg, y_pred_pos[i]))
    
    # 计算 V01 = E[psi(X, Y) | X=0]
    v01 = np.zeros(n0)
    for i in range(n0):
        v01[i] = np.mean(psi(y_pred_neg[i], y_pred_pos))
    
    # DeLong 方差估计
    s10 = np.var(v10) / n1 if n1 > 1 else 0
    s01 = np.var(v01) / n0 if n0 > 1 else 0
    var_auc = s10 + s01
    return var_auc, 1/np.sqrt(var_auc) if var_auc > 0 else 0

def delong_covariance(y_true, pred1, pred2):
    """计算两个模型 AUC 的 DeLong 协方差"""
    n = len(y_true)
    n1 = np.sum(y_true)
    n0 = n - n1
    if n1 < 2 or n0 < 2:
        return 0.0
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    def psi(a, b):
        return np.where(a < b, 1.0, np.where(a == b, 0.5, 0.0))
    
    # 估计协方差分量
    cov_10 = 0
    cov_01 = 0
    for i in range(n1):
        psi1 = np.mean(psi(pred2[neg_idx], pred1[pos_idx[i]]))
        psi2 = np.mean(psi(pred1[neg_idx], pred2[pos_idx[i]]))
        cov_10 += (np.mean(psi(pred1[neg_idx], pred1[pos_idx[i]])) - psi1) * \
                  (np.mean(psi(pred2[neg_idx], pred2[pos_idx[i]])) - psi2)
    cov_10 /= n1
    
    for i in range(n0):
        psi1 = np.mean(psi(pred2[neg_idx[i]], pred1[pos_idx]))
        psi2 = np.mean(psi(pred1[neg_idx[i]], pred2[pos_idx]))
        cov_01 += (np.mean(psi(pred1[neg_idx[i]], pred1[pos_idx])) - psi1) * \
                  (np.mean(psi(pred2[neg_idx[i]], pred2[pos_idx])) - psi2)
    cov_01 /= n0
    
    return cov_10/n1 + cov_01/n0

def delong_roc_test(y_true, pred1, pred2):
    """DeLong 检验：比较两个 AUC 差异"""
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    
    var1, _ = delong_roc_variance(y_true, pred1)
    var2, _ = delong_roc_variance(y_true, pred2)
    cov12 = delong_covariance(y_true, pred1, pred2)
    
    diff = auc1 - auc2
    var_diff = var1 + var2 - 2*cov12
    
    if var_diff <= 0:
        return abs(diff), 0.5 if diff == 0 else 0.0
    
    z = diff / np.sqrt(var_diff)
    p = 2 * (1 - norm.cdf(abs(z)))  # 双边检验
    return abs(diff), p

# 执行检验
models = {
    '随机森林': rf_proba,
    'Logistic回归': lr_proba,
    'XGBoost': xgb_proba
}

print("\nDeLong检验结果（手动实现）：")
for name1, pred1 in models.items():
    for name2, pred2 in models.items():
        if name1 >= name2:
            continue
        diff, p = delong_roc_test(y_test, pred1, pred2)
        auc1 = roc_auc_score(y_test, pred1)
        auc2 = roc_auc_score(y_test, pred2)
        print(f"  {name1} vs {name2}: AUC差值={diff:.4f}, p={p:.6f}")
print("\n" + "=" * 60)
print("第三章 运行完成！")
print(f"所有结果已保存至: {RESULTS_DIR}")
print("=" * 60)