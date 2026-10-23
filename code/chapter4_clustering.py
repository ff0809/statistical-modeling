"""
第四章：癌症代谢亚型分析
对应写作框架：4.1 模型建立（PCA降维 + K-Means聚类）
              4.2 实验设计
              4.3 结果分析
              4.4 模型性能分析

用法：python code/chapter4_clustering.py
输出：results/chapter4/ 下的表格和图表
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# ============================================================
# 路径设置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'chapter4')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("  第四章：癌症代谢亚型分析")
print("  基于代谢组学的癌症预测模型及其可解释性研究")
print("=" * 80)

# ============================================================
# 数据加载与预处理（与第三章一致）
# ============================================================
print("\n" + "-" * 60)
print("数据加载与预处理")
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

# 构建特征矩阵
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

print(f"特征矩阵: {X.shape[0]} 个队列 × {X.shape[1]} 个代谢物特征")
print(f"癌症类型分布:")
for c in top_cancers:
    print(f"  {c}: {(y_labels == c).sum()}")

# Z-score 标准化（与第三章一致）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 4.1.1 PCA 主成分分析
# ============================================================
print("\n" + "=" * 60)
print("4.1.1 PCA 降维")
print("=" * 60)

# 执行 PCA（保留足够成分以计算累积方差）
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# 方差解释表
explained_var_ratio = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var_ratio)

pca_var_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(len(explained_var_ratio))],
    'Eigenvalue': pca_full.explained_variance_,
    'Explained_Variance_Ratio': explained_var_ratio,
    'Cumulative_Ratio': cumulative_var
})
pca_var_df['Explained_Variance_Ratio'] = pca_var_df['Explained_Variance_Ratio'] * 100
pca_var_df['Cumulative_Ratio'] = pca_var_df['Cumulative_Ratio'] * 100

# 保存前10个主成分
pca_var_df.head(10).to_csv(os.path.join(RESULTS_DIR, 'pca_variance.csv'), index=False)

print("\n主成分方差解释 (前5个):")
print(pca_var_df.head(5).to_string(index=False))
print(f"\n前2个主成分累计解释: {cumulative_var[1]*100:.1f}%")
print(f"前4个主成分累计解释: {cumulative_var[3]*100:.1f}%")

# PCA 降至2维用于可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pc1_var, pc2_var = pca.explained_variance_ratio_

print(f"PC1 方差解释率: {pc1_var*100:.1f}%")
print(f"PC2 方差解释率: {pc2_var*100:.1f}%")

# ---- 图1: PCA 方差解释率（Scree Plot） ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：Scree Plot
ax = axes[0]
n_show = min(15, len(explained_var_ratio))
ax.bar(range(1, n_show+1), explained_var_ratio[:n_show]*100,
       color='steelblue', alpha=0.7, edgecolor='black')
ax.plot(range(1, n_show+1), cumulative_var[:n_show]*100,
        'ro-', markersize=6, linewidth=2, label='Cumulative')
ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Principal Component', fontsize=11)
ax.set_ylabel('Variance Explained (%)', fontsize=11)
ax.set_title('Scree Plot', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 右图：方差解释饼图
ax = axes[1]
sizes = [pc1_var*100, pc2_var*100, max(0, 100 - (pc1_var+pc2_var)*100)]
labels = [f'PC1 ({sizes[0]:.1f}%)', f'PC2 ({sizes[1]:.1f}%)', 'Remaining']
colors_pie = ['#1f77b4', '#ff7f0e', '#d3d3d3']
ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
       startangle=90, explode=(0.05, 0.05, 0))
ax.set_title('PCA Variance Decomposition', fontsize=13, fontweight='bold')

plt.suptitle('PCA 主成分分析 — 方差解释', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pca_variance_analysis.png'), dpi=200, bbox_inches='tight')
print("已保存: pca_variance_analysis.png")

# ---- 图2: PCA 散点图（按癌症类型着色） ----
fig, ax = plt.subplots(figsize=(10, 8))
cancer_types = np.unique(y_labels)
colors_map = plt.cm.Set1(np.linspace(0, 1, len(cancer_types)))

for i, cancer in enumerate(cancer_types):
    mask = y_labels == cancer
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[colors_map[i]], label=cancer.replace(' cancer', ''),
               alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

ax.set_xlabel(f'PC1 ({pc1_var*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pc2_var*100:.1f}%)', fontsize=12)
ax.set_title('PCA Scatter Plot — Five Cancer Types in Metabolic Space',
             fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pca_scatter_by_cancer.png'), dpi=200, bbox_inches='tight')
print("已保存: pca_scatter_by_cancer.png")

# ============================================================
# 4.1.2 K-Means 聚类 + 4.1.3 评估
# ============================================================
print("\n" + "=" * 60)
print("4.1.2 K-Means 聚类 + 4.1.3 轮廓系数评估")
print("=" * 60)

# ---- 肘部法则: 确定最优 k ----
k_range = range(2, 11)
wcss_list = []
silhouette_list = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    wcss_list.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouette_list.append(sil)
    print(f"  k={k}: WCSS={km.inertia_:.2f}, Silhouette={sil:.4f}")

# ---- 图3: 肘部法则 + 轮廓系数 ----
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'steelblue'
ax1.plot(list(k_range), wcss_list, 'o-', color=color1, linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color2 = 'coral'
ax2.plot(list(k_range), silhouette_list, 's--', color=color2, linewidth=2, markersize=8)
ax2.set_ylabel('Silhouette Score', fontsize=12, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# 标注最优 k (取轮廓系数最高且肘部拐点的 k)
best_k = list(k_range)[np.argmax(silhouette_list)]
ax1.axvline(x=best_k, color='gray', linestyle='--', alpha=0.5)
ax1.text(best_k + 0.1, ax1.get_ylim()[1] * 0.9,
         f'Optimal k={best_k}', fontsize=12, color='red', fontweight='bold')

plt.title('Elbow Method & Silhouette Analysis — Optimal k Selection',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'elbow_silhouette.png'), dpi=200, bbox_inches='tight')
print(f"已保存: elbow_silhouette.png")
print(f"\n最优 k = {best_k} (基于轮廓系数)")

# ============================================================
# 4.2.3 最终聚类结果 (k=5)
# ============================================================
print("\n" + "=" * 60)
print("4.2.3 最终聚类 (k=5)")
print("=" * 60)

k_final = 5
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
final_silhouette = silhouette_score(X_scaled, clusters)

print(f"最终聚类 k={k_final}, 轮廓系数={final_silhouette:.4f}")

# 聚类 vs 癌症类型的交叉表
cluster_df = pd.DataFrame({
    'Cancer_Type': y_labels,
    'Cancer_Type_Short': y_labels.str.replace(' cancer', ''),
    'Cluster': [f'C{i}' for i in clusters]
})

crosstab = pd.crosstab(cluster_df['Cancer_Type_Short'], cluster_df['Cluster'])
print("\n聚类与癌症类型交叉表:")
print(crosstab)

# 保存交叉表
crosstab.to_csv(os.path.join(RESULTS_DIR, 'cluster_vs_cancer_crosstab.csv'))

# ---- 图4: PCA 空间聚类可视化（左右对比） ----
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 左图：按实际癌种着色
ax = axes[0]
for i, cancer in enumerate(cancer_types):
    mask = y_labels == cancer
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[colors_map[i]], label=cancer.replace(' cancer', ''),
               alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pc1_var*100:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({pc2_var*100:.1f}%)', fontsize=11)
ax.set_title('(a) Colored by Actual Cancer Type', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

# 右图：按聚类着色
ax = axes[1]
cluster_color_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i in range(k_final):
    mask = clusters == i
    label_counts = pd.Series(clusters)[mask].value_counts()
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[cluster_color_map[i]], label=f'C{i}',
               alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pc1_var*100:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({pc2_var*100:.1f}%)', fontsize=11)
ax.set_title('(b) Colored by K-Means Cluster', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('PCA Space Clustering Visualization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pca_clustering_comparison.png'), dpi=200, bbox_inches='tight')
print("已保存: pca_clustering_comparison.png")

# ============================================================
# 4.2.4 亚型代谢标志物分析
# ============================================================
print("\n" + "-" * 60)
print("4.2.4 各亚型代谢标志物分析")
print("-" * 60)

# 计算每个簇的代谢物均值
X_cluster = pd.DataFrame(X_scaled, columns=feature_names)
X_cluster['Cluster'] = clusters

cluster_means = X_cluster.groupby('Cluster').mean()
cluster_markers = []

for c in range(k_final):
    # 提取每个簇中平均 log2FC 绝对值最大的前5个代谢物
    cluster_row = cluster_means.loc[c].sort_values(key=abs, ascending=False)
    top5 = cluster_row.head(5)

    top_cancer = cluster_df[cluster_df['Cluster'] == f'C{c}']['Cancer_Type_Short'] \
                           .value_counts().index[0]

    markers = []
    for meta, val in top5.items():
        direction = '↑' if val > 0 else '↓'
        markers.append(f'{meta}{direction}')

    cluster_markers.append({
        'Cluster': f'C{c}',
        'Main_Cancer_Type': top_cancer,
        'Top_Metabolites': '; '.join(markers),
        'Score': final_silhouette
    })
    print(f"  C{c} ({top_cancer}): {' | '.join(markers)}")

cluster_markers_df = pd.DataFrame(cluster_markers)
cluster_markers_df.to_csv(os.path.join(RESULTS_DIR, 'cluster_markers.csv'), index=False)

# ---- 图5: 各亚型特征代谢物热图 ----
fig, ax = plt.subplots(figsize=(12, 9))

# 取每个簇前5个最重要代谢物的并集
top_metabolites = set()
for c in range(k_final):
    cluster_row = cluster_means.loc[c].sort_values(key=abs, ascending=False)
    top_metabolites.update(cluster_row.head(5).index.tolist())

heatmap_data = cluster_means[list(top_metabolites)].T

sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, annot=True,
            fmt='.2f', linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Mean log2FC (Z-score normalized)'})
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Metabolite', fontsize=12)
ax.set_title('Cluster-Specific Metabolic Marker Profiles',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'cluster_marker_heatmap.png'),
            dpi=200, bbox_inches='tight')
print("已保存: cluster_marker_heatmap.png")

# ============================================================
# 4.4 模型性能评估
# ============================================================
print("\n" + "=" * 60)
print("4.4 模型性能评估")
print("=" * 60)

from sklearn.preprocessing import LabelEncoder

# ARI (调整兰德指数)
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
ari_score = adjusted_rand_score(y_encoded, clusters)

print(f"轮廓系数 (Silhouette Score): {final_silhouette:.4f}")
print(f"调整兰德指数 (ARI): {ari_score:.4f}")

perf_df = pd.DataFrame({
    'Metric': ['Silhouette Score', 'ARI', 'Number of Clusters'],
    'Value': [f'{final_silhouette:.4f}', f'{ari_score:.4f}', k_final]
})
perf_df.to_csv(os.path.join(RESULTS_DIR, 'clustering_performance.csv'), index=False)

# 保存所有数值结果到汇总文本文件
with open(os.path.join(RESULTS_DIR, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(f"第四章 分析结果汇总\n")
    f.write(f"=" * 40 + "\n")
    f.write(f"PCA:\n")
    f.write(f"  PC1: {pc1_var*100:.1f}%\n")
    f.write(f"  PC2: {pc2_var*100:.1f}%\n")
    f.write(f"  PC1+PC2累计: {(pc1_var+pc2_var)*100:.1f}%\n\n")
    f.write(f"聚类:\n")
    f.write(f"  最优 k: {k_final}\n")
    f.write(f"  轮廓系数: {final_silhouette:.4f}\n")
    f.write(f"  ARI: {ari_score:.4f}\n")

print(f"\n结果汇总已保存至: {os.path.join(RESULTS_DIR, 'analysis_summary.txt')}")
print("\n" + "=" * 60)
print("第四章 运行完成！")
print(f"所有结果已保存至: {RESULTS_DIR}")
print("=" * 60)
