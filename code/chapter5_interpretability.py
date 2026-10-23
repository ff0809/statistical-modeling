"""
第五章：特征代谢物识别与依赖路径解释
对应写作框架：5.1 模型建立（特征融合 + 贝叶斯网络深化 + 结构-统计融合）
              5.2 实验设计（一致性分析 + 消融实验 + KEGG比对）
              5.3 结果分析
              5.4 模型性能分析

用法：python code/chapter5_interpretability.py
      （会在内部调用第三章模型训练，自动提取特征重要性）
输出：results/chapter5/ 下的表格和图表
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
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import spearmanr
import xgboost as xgb

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# ============================================================
# 路径设置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'chapter5')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("  第五章：特征代谢物识别与依赖路径解释")
print("  基于代谢组学的癌症预测模型及其可解释性研究")
print("=" * 80)

# ============================================================
# 数据预处理（复用第三章逻辑）
# ============================================================
print("\n" + "-" * 60)
print("数据预处理")
print("-" * 60)

metabolite_df = pd.read_csv(os.path.join(DATA_DIR, 'downloads.metabolite.txt'), sep='\t')
study_df = pd.read_csv(os.path.join(DATA_DIR, 'downloads.study.txt'), sep='\t')

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
feature_names = feature_matrix.columns.tolist()

print(f"特征矩阵: {X.shape[0]} 样本 × {X.shape[1]} 特征")

y_binary = (y_labels == 'breast cancer').astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# 先切分再标准化：防止测试集数据泄漏到标准化参数中
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# 保留全量标准化后的矩阵，供贝叶斯网络和消融实验使用
X_scaled = scaler.transform(X)

# ============================================================
# 5.1.1: 训练三个模型 + 提取特征重要性
# ============================================================
print("\n" + "=" * 60)
print("5.1.1 多模型特征重要性提取")
print("=" * 60)

# ---- Logistic Regression ----
lr_model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
lr_model.fit(X_train, y_train)

lr_importance = pd.DataFrame({
    'metabolite': feature_names,
    'lr_importance': np.abs(lr_model.coef_[0])  # 权重绝对值
})
lr_importance['lr_rank'] = lr_importance['lr_importance'].rank(ascending=False)

# ---- Random Forest ----
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_features='sqrt')
rf_model.fit(X_train, y_train)

rf_importance = pd.DataFrame({
    'metabolite': feature_names,
    'rf_importance': rf_model.feature_importances_
})
rf_importance['rf_rank'] = rf_importance['rf_importance'].rank(ascending=False)

# ---- XGBoost ----
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

xgb_importance = pd.DataFrame({
    'metabolite': feature_names,
    'xgb_importance': xgb_model.feature_importances_
})
xgb_importance['xgb_rank'] = xgb_importance['xgb_importance'].rank(ascending=False)

# ---- 融合特征重要性 ----
# Min-Max 归一化 (公式: Importance_norm = (x - min) / (max - min))
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

fusion_df = lr_importance[['metabolite']].copy()
fusion_df['LR_Importance'] = min_max_normalize(lr_importance['lr_importance'])
fusion_df['RF_Importance'] = min_max_normalize(rf_importance['rf_importance'])
fusion_df['XGB_Importance'] = min_max_normalize(xgb_importance['xgb_importance'])
fusion_df['Mean_Importance'] = fusion_df[['LR_Importance', 'RF_Importance', 'XGB_Importance']].mean(axis=1)
fusion_df['LR_Rank'] = lr_importance['lr_rank'].astype(int)
fusion_df['RF_Rank'] = rf_importance['rf_rank'].astype(int)
fusion_df['XGB_Rank'] = xgb_importance['xgb_rank'].astype(int)
fusion_df['Mean_Rank'] = fusion_df[['LR_Rank', 'RF_Rank', 'XGB_Rank']].mean(axis=1)

fusion_df = fusion_df.sort_values('Mean_Importance', ascending=False)
fusion_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance_fusion.csv'), index=False)

# ============================================================
# 5.2.2: 一致性分析（Spearman相关系数）
# ============================================================
print("\n" + "-" * 60)
print("5.2.2 模型间特征重要性一致性分析")
print("-" * 60)

pairwise_results = []
pairs = [
    ('LR vs RF', lr_importance['lr_importance'], rf_importance['rf_importance']),
    ('LR vs XGB', lr_importance['lr_importance'], xgb_importance['xgb_importance']),
    ('RF vs XGB', rf_importance['rf_importance'], xgb_importance['xgb_importance']),
]

spearman_results = []
for label, a, b in pairs:
    rho, pval = spearmanr(a, b)
    spearman_results.append({'Pair': label, 'Spearman_rho': rho, 'p_value': pval})
    sig = '显著' if pval < 0.01 else ('较显著' if pval < 0.05 else '不显著')
    print(f"  {label}: ρ={rho:.4f}, p={pval:.6f} ({sig})")

spearman_df = pd.DataFrame(spearman_results)
spearman_df.to_csv(os.path.join(RESULTS_DIR, 'spearman_correlation.csv'), index=False)

# ============================================================
# 5.3.2: 一致性 Top 关键代谢物
# ============================================================
print("\n" + "-" * 60)
print("5.3.2 一致性 Top 关键代谢物")
print("-" * 60)

# 筛选条件：在三个模型中均排名 Top 30
top30_lr = set(lr_importance.nsmallest(30, 'lr_rank')['metabolite'])
top30_rf = set(rf_importance.nsmallest(30, 'rf_rank')['metabolite'])
top30_xgb = set(xgb_importance.nsmallest(30, 'xgb_rank')['metabolite'])
consistent_top = top30_lr & top30_rf & top30_xgb

print(f"三模型均排名 Top 30 的代谢物数: {len(consistent_top)}")

# 详细 Top 15 表
top15 = fusion_df.head(15).copy()

# 添加 log2FC 方向和跨研究数
original_values = valid_data.groupby('original_metabolite_name').agg({
    'log2FC': 'mean',
    'Cohort_id': 'nunique'
}).rename(columns={'log2FC': 'mean_log2FC', 'Cohort_id': 'n_studies'})

top15 = top15.merge(original_values, left_on='metabolite', right_index=True, how='left')
top15['Regulation'] = top15['mean_log2FC'].apply(lambda x: '上调' if x > 0 else '下调')
top15.to_csv(os.path.join(RESULTS_DIR, 'consistent_top_metabolites.csv'), index=False)

print("\nTop 15 一致性关键代谢物:")
print(top15[['metabolite', 'Mean_Importance', 'Mean_Rank', 'mean_log2FC', 'Regulation']].head(15).to_string(index=False))

# ---- 图1: 三模型特征重要性一致性散点图 ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

pair_data = [
    ('LR', 'RF', lr_importance['lr_importance'], rf_importance['rf_importance']),
    ('LR', 'XGB', lr_importance['lr_importance'], xgb_importance['xgb_importance']),
    ('RF', 'XGB', rf_importance['rf_importance'], xgb_importance['xgb_importance']),
]
for idx, (name1, name2, imp1, imp2) in enumerate(pair_data):
    ax = axes[idx]
    rho, _ = spearmanr(imp1, imp2)
    ax.scatter(imp1, imp2, alpha=0.5, s=20, c='steelblue')
    ax.set_xlabel(f'{name1} Importance', fontsize=10)
    ax.set_ylabel(f'{name2} Importance', fontsize=10)
    ax.set_title(f'{name1} vs {name2} (ρ={rho:.3f})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Cross-Model Feature Importance Consistency', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_consistency.png'), dpi=200, bbox_inches='tight')
print("已保存: feature_importance_consistency.png")

# ---- 图2: Top 20 融合重要性柱状图 ----
fig, ax = plt.subplots(figsize=(12, 9))
top20 = fusion_df.head(20)
x_pos = np.arange(len(top20))
width = 0.25

ax.bar(x_pos - width, top20['LR_Importance'], width, label='LR', color='#1f77b4', alpha=0.85)
ax.bar(x_pos, top20['RF_Importance'], width, label='RF', color='#ff7f0e', alpha=0.85)
ax.bar(x_pos + width, top20['XGB_Importance'], width, label='XGBoost', color='#2ca02c', alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels(top20['metabolite'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Normalized Importance', fontsize=11)
ax.set_title('Top 20 Metabolites — Multi-Model Importance Comparison',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'top20_fused_importance.png'), dpi=200, bbox_inches='tight')
print("已保存: top20_fused_importance.png")

# ============================================================
# 5.1.2 + 5.2.4: 贝叶斯网络依赖路径分析
# ============================================================
print("\n" + "-" * 60)
print("5.1.2 / 5.2.4 贝叶斯网络依赖路径分析")
print("-" * 60)

try:
    from pgmpy.estimators import HillClimbSearch, BicScore

    # 构建离散化数据
    X_df = pd.DataFrame(X, columns=feature_names)
    X_discrete = X_df.copy()

    meta_freq = feature_pivot.notna().sum().sort_values(ascending=False)
    top28 = meta_freq.head(28).index.tolist()
    X_discrete = X_discrete[top28]
    X_discrete['Cancer_Status'] = y_binary.values

    for col in X_discrete.columns:
        if col == 'Cancer_Status':
            continue
        X_discrete[col] = pd.cut(
            X_discrete[col],
            bins=[-float('inf'), -1, 1, float('inf')],
            labels=[0, 1, 2]
        ).astype(int)

    X_discrete['Cancer_Status'] = X_discrete['Cancer_Status'].astype(int)

    # Hill-Climbing 结构学习
    hc = HillClimbSearch(X_discrete, scoring_method=BicScore(X_discrete))
    dag = hc.estimate(max_indegree=5, max_iter=2000)

    if len(dag.edges()) > 0:
        # ---- 马尔可夫 Blanket ----
        mb = list(dag.get_markov_blanket('Cancer_Status'))
        print(f"癌症状态 Y 的马尔可夫 Blanket: {mb}")

        # ---- PageRank 中心性计算 ----
        import networkx as nx
        G = nx.DiGraph(dag.edges())
        pagerank = nx.pagerank(G, alpha=0.85)
        pagerank_df = pd.DataFrame(
            list(pagerank.items()),
            columns=['metabolite', 'PageRank']
        ).sort_values('PageRank', ascending=False)
        pagerank_df.to_csv(os.path.join(RESULTS_DIR, 'pagerank_centrality.csv'), index=False)

        print("\nPageRank Top 5 中心节点:")
        print(pagerank_df.head(5).to_string(index=False))

        # 边列表
        edges_list = list(dag.edges())
        edges_df = pd.DataFrame(edges_list, columns=['source', 'target'])
        edges_df.to_csv(os.path.join(RESULTS_DIR, 'bayesian_network_edges.csv'), index=False)

        # 标记 MB 中的边
        edges_df['in_MB'] = edges_df.apply(
            lambda r: r['source'] in mb or r['target'] in mb, axis=1
        )

        # ---- 绘制 DAG（标注 MB）----
        pos = nx.spring_layout(G, seed=42, k=3, iterations=50)

        fig, ax = plt.subplots(figsize=(16, 12))
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node == 'Cancer_Status':
                node_colors.append('red')
                node_sizes.append(1500)
            elif node in mb:
                node_colors.append('gold')
                node_sizes.append(1000)
            else:
                node_colors.append('lightblue')
                node_sizes.append(600)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.85)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20,
                               edge_color='gray', alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8)

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=12, label='Cancer Status (Y)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                       markersize=10, label='Markov Blanket'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                       markersize=8, label='Other Metabolites'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.title('Bayesian Network DAG — Markov Blanket Highlighted',
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'bayesian_network_dag.png'),
                   dpi=200, bbox_inches='tight')
        print("已保存: bayesian_network_dag.png")

        # MB 代谢物列表
        mb_df = pd.DataFrame({'metabolite': mb})
        mb_df.to_csv(os.path.join(RESULTS_DIR, 'markov_blanket.csv'), index=False)
    else:
        print("注意: 贝叶斯网络未学习到有效边")
        mb = []

except ImportError:
    print("pgmpy 或 networkx 未安装，跳过贝叶斯网络分析")
    print("安装命令: pip install pgmpy networkx")
    mb = []

# ============================================================
# 5.2.5: 消融实验
# ============================================================
print("\n" + "-" * 60)
print("5.2.5 消融实验 — 马尔可夫 Blanket 验证")
print("-" * 60)

# 用 MB(Y) 代谢物子集训练 XGBoost
if len(mb) > 0:
    # 找到 MB(Y) 在特征矩阵中的索引
    mb_available = [m for m in mb if m in feature_names]
    print(f"MB(Y) 中可用代谢物: {mb_available}")

    if len(mb_available) >= 2:
        mb_indices = [feature_names.index(m) for m in mb_available]
        X_mb = X_scaled[:, mb_indices]

        X_mb_train, X_mb_test, _, _ = train_test_split(
            X_mb, y_binary, test_size=0.3, random_state=42, stratify=y_binary
        )

        xgb_mb = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        xgb_mb.fit(X_mb_train, y_train)
        xgb_mb_proba = xgb_mb.predict_proba(X_mb_test)[:, 1]
        xgb_mb_auc = auc(*roc_curve(y_test, xgb_mb_proba)[:2])

        # 全特征模型（复用第三章训练好的模型）
        xgb_full_proba = xgb_model.predict_proba(X_test)[:, 1]
        xgb_full_auc = auc(*roc_curve(y_test, xgb_full_proba)[:2])

        ablation_df = pd.DataFrame({
            'Model': [
                f'XGBoost (Full: {len(feature_names)} features)',
                f'XGBoost (MB only: {len(mb_available)} metabolites)'
            ],
            'AUC': [xgb_full_auc, xgb_mb_auc],
            'Feature_Count': [len(feature_names), len(mb_available)],
            'Performance_Ratio': [1.0, xgb_mb_auc / xgb_full_auc]
        })
        ablation_df.to_csv(os.path.join(RESULTS_DIR, 'ablation_results.csv'), index=False)

        print(f"\n消融实验结果:")
        print(ablation_df.to_string(index=False))
        print(f"\n性能保持率: {xgb_mb_auc / xgb_full_auc * 100:.1f}%")

        # ---- 消融实验 ROC 对比图 ----
        fig, ax = plt.subplots(figsize=(9, 7))
        fpr_full, tpr_full, _ = roc_curve(y_test, xgb_full_proba)
        fpr_mb, tpr_mb, _ = roc_curve(y_test, xgb_mb_proba)
        ax.plot(fpr_full, tpr_full, 'g-', lw=2, label=f'Full Features (AUC={xgb_full_auc:.4f})')
        ax.plot(fpr_mb, tpr_mb, 'orange', lw=2, ls='--',
                label=f'MB Only — {len(mb_available)} metabolites (AUC={xgb_mb_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'gray', lw=1, ls='--', alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Ablation Study — Full Features vs Markov Blanket Only',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'ablation_roc_comparison.png'),
                   dpi=200, bbox_inches='tight')
        print("已保存: ablation_roc_comparison.png")
    else:
        print("MB(Y) 可用代谢物少于2个，跳过消融实验")
else:
    print("无马尔可夫 Blanket 结果，跳过消融实验")

# ============================================================
# 5.2.6: KEGG 通路比对
# ============================================================
print("\n" + "-" * 60)
print("5.2.6 KEGG 通路比对验证")
print("-" * 60)

# 已知 KEGG 代谢通路参考表
kegg_reference = pd.DataFrame({
    'source': ['Glucose', 'Glucose', 'Glutamine', 'Glutamine',
               'Glutamine', 'PC(18:1)', 'Lactate', 'Lactate'],
    'target': ['Lactate', 'Pyruvate', 'Glutamate', 'Alanine',
               'Proline', 'Lysophospholipid', 'Pyruvate', 'H+'],
    'pathway': ['Glycolysis', 'Glycolysis', 'Glutamate metabolism',
                'Alanine metabolism', 'Proline metabolism',
                'Glycerophospholipid metabolism',
                'Gluconeogenesis', 'Tumor microenvironment'],
    'kegg_id': ['map00010', 'map00010', 'map00250',
                'map00270', 'map00330', 'map00564',
                'map00010', 'map04936'],
    'evidence': ['Warburg effect', 'Warburg effect', 'Cell, 2011',
                 'Nature, 2015', 'Cancer Res, 2018',
                 'Nature, 2020', 'Cell Metab, 2016', 'Cancer Res, 2019']
})

# 从贝叶斯网络边中匹配 KEGG 通路
if len(dag.edges()) > 0 if 'dag' in dir() else False:
    # 计算匹配数量
    matches = []
    for _, row in kegg_reference.iterrows():
        match = any(
            (s.lower() in e[0].lower() or e[0].lower() in s.lower()) and
            (t.lower() in e[1].lower() or e[1].lower() in t.lower())
            for s, t in [(row['source'], row['target'])]
            for e in dag.edges()
        )
        matches.append({
            'source': row['source'],
            'target': row['target'],
            'pathway': row['pathway'],
            'kegg_id': row['kegg_id'],
            'matched_in_bn': 'Yes' if match else 'No'
        })

    kegg_df = pd.DataFrame(matches)
    kegg_df.to_csv(os.path.join(RESULTS_DIR, 'kegg_pathway_comparison.csv'), index=False)

    matched_count = sum(1 for m in matches if m['matched_in_bn'] == 'Yes')
    total = len(matches)
    print(f"KEGG 通路匹配率: {matched_count}/{total} ({matched_count/total*100:.1f}%)")
    print("\nKEGG 比对结果:")
    print(kegg_df.to_string(index=False))
else:
    # 保存参考 KEGG 通路表
    kegg_reference.to_csv(os.path.join(RESULTS_DIR, 'kegg_reference_pathways.csv'), index=False)
    print("KEGG 参考通路表已保存")
    print("（贝叶斯网络未生成有效边，KEGG 比对需要在运行贝叶斯网络后才可展开）")

# ============================================================
# 5.4.1: 结构-统计融合模型性能
# ============================================================
print("\n" + "=" * 60)
print("5.4 结构-统计融合框架性能分析")
print("=" * 60)

if len(mb) > 0 and len(mb_available) >= 2:
    fusion_perf = pd.DataFrame({
        'Model': [
            f'XGBoost (Full Features)',
            f'XGBoost + BN Structure (MB only)'
        ],
        'Feature_Count': [len(feature_names), len(mb_available)],
        'AUC': [f'{xgb_full_auc:.4f}', f'{xgb_mb_auc:.4f}'],
        'Dimension_Reduction': ['-', f'{(1 - len(mb_available)/len(feature_names))*100:.0f}%'],
    })
    fusion_perf.to_csv(os.path.join(RESULTS_DIR, 'fusion_model_performance.csv'), index=False)
    print("\n结构-统计融合模型性能:")
    print(fusion_perf.to_string(index=False))

# ============================================================
# 汇总保存
# ============================================================
summary_lines = [
    "第五章：特征代谢物识别与依赖路径解释 — 分析结果汇总",
    "=" * 50,
    "",
    "5.3.1 多模型特征重要性一致性",
    "-" * 30,
]
for r in spearman_results:
    summary_lines.append(f"  {r['Pair']}: ρ={r['Spearman_rho']:.4f}, p={r['p_value']:.6f}")

summary_lines.extend([
    "",
    "5.3.2 一致性 Top 代谢物",
    "-" * 30,
    f"  三模型均排 Top 30 数量: {len(consistent_top)}",
    "",
    "5.3.3 贝叶斯网络拓扑",
    "-" * 30,
])
if len(mb) > 0:
    summary_lines.append(f"  马尔可夫 Blanket: {mb}")

if len(mb_available) >= 2:
    summary_lines.extend([
        "",
        "5.4.1 消融实验",
        "-" * 30,
        f"  全特征 AUC: {xgb_full_auc:.4f}",
        f"  MB Only AUC: {xgb_mb_auc:.4f}",
        f"  性能保持率: {xgb_mb_auc / xgb_full_auc * 100:.1f}%",
    ])

with open(os.path.join(RESULTS_DIR, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print("\n" + "=" * 60)
print("第五章 运行完成！")
print(f"所有结果已保存至: {RESULTS_DIR}")
print("=" * 60)
