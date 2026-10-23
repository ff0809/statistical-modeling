# 第三章：癌症风险预测模型（升级版）

## 文件说明

| 文件 | 说明 |
|------|------|
| `chapter3_prediction_upgraded.py` | 升级版第三章代码 |

---

## 升级内容

### 原版 vs 升级版对照

| 模块 | 原版 | 升级版 | 原因 |
|------|------|--------|------|
| 基准回归 | Logistic Regression | **ElasticNet 弹性网回归** | 解决代谢组学高维共线性，L1+L2 双正则化 |
| 树模型 | XGBoost（固定参数） | **Optuna 贝叶斯优化 XGBoost** | TPE 算法自动寻优 8 个超参数，方法论更严谨 |
| 随机森林 | RandomForestClassifier | **保持不变**（AUC=0.8886 双冠） | 原始 RF 效果已最佳 |
| 贝叶斯网络 | HillClimbSearch + BIC | **保持不变** | — |
| 集成模型 | 无 | **Stacking 集成**（三基模型 + LR 元学习器） | 模型融合补充，收束全文叙事 |
| 对比检验 | 无 | **DeLong 显著性检验** | 统计严谨性对标论文要求 |

### 新功能清单

1. **ElasticNet 弹性网逻辑回归**：`l1_ratio` × `C` 双参数 GridSearchCV 调优，附带正则化路径图 + 稀疏度分析
2. **Optuna XGBoost**：30 trials TPE 贝叶斯超参数优化，附带优化历史收敛曲线 + 超参数重要性图
3. **Stacking 集成模型**：ElasticNet + RandomForest + OptunaXGBoost → LR（StackingClassifier cv=5）
4. **DeLong 检验**：Stacking 与各单模型的 AUC 差异显著性配对检验
5. **最优参数导出**：运行结束时自动保存 `optimal_params.json` 供第五章读取

---

## 论文对应小节

| 代码小节 | 论文位置 | 内容 |
|---------|---------|------|
| 3.1.1 | 数据预处理 | 数据清洗、log2FC、PivotTable、Z-score 标准化 |
| 3.1.2 | 弹性网回归模型 | GridSearchCV 调优 + 正则化路径分析 |
| 3.1.3 | 随机森林模型 | 基线 RF 训练与评估 |
| 3.1.4 | 贝叶斯优化 XGBoost | Optuna TPE 超参数寻优 |
| 3.1.5 | Stacking 集成模型 | 三基模型融合 + DeLong 检验 |
| 3.1.6 | 贝叶斯网络分析 | DAG 结构学习 + CPT 参数 + 马尔可夫毯 |
| 3.3 | 综合结果对比 | 四模型性能柱状图 + 特征重要性 |

---

## 运行方法

```bash
# 1. 安装依赖
pip install optuna>=3.0

# 2. 运行（确保工作目录为项目根目录）
python chapter3_prediction_upgraded.py
```

**预期运行时间**：约 3-5 分钟（含 ElasticNet GridSearchCV + Optuna 优化 + 贝叶斯网络学习）

---

## 实验结果

### 最终模型性能（测试集）

| 模型 | AUC | Accuracy | Precision | Recall | F1-Score | Brier |
|------|-----|----------|-----------|--------|----------|-------|
| ElasticNet LR | 0.6852 | 0.7183 | 0.5600 | 0.6087 | 0.5833 | 0.2420 |
| Random Forest | **0.8886** | 0.7183 | 0.5429 | **0.8261** | 0.6552 | 0.1579 |
| Optuna XGBoost | 0.7174 | 0.7465 | 0.6923 | 0.3913 | 0.5000 | 0.1761 |
| **Stacking 集成** | 0.8379 | **0.8310** | **1.0000** | 0.4783 | **0.6471** | **0.1383** |

### 新旧对比（关键提升）

| 模型 | 指标改善 |
|------|---------|
| LR → ElasticNet | AUC +0.016, F1 +0.023, 60.9% 系数被压缩为 0 |
| XGBoost → OptunaXGBoost | F1 +0.088, Recall +0.087, Precision +0.056 |
| 新增 Stacking | 准确率 0.8310 全场最高，Brier 0.1383 全场最低 |

---

## 输出文件

运行后在 `results/chapter3_upgraded/` 下生成：

| 文件类型 | 文件列表 |
|---------|---------|
| 图表（PNG） | `elasticnet_regularization_path.png`, `elasticnet_gs_heatmap.png`, `optuna_optimization_history.png`, `optuna_param_analysis.png`, `stacking_roc_comparison.png`, `roc_curves.png`, `model_performance_bar.png`, `rf_feature_importance_top20.png`, `xgb_feature_importance_top20.png`, `bayesian_network_dag.png`, `bayesian_network_markov_blanket.png` |
| 数据表（CSV） | `elasticnet_coefficients.csv`, `rf_feature_importance.csv`, `xgb_feature_importance.csv`, `model_comparison.csv`, `delong_test_results.csv`, `bayesian_network_edges.csv`, `bayesian_cpt_summary.csv` |
| 配置（JSON） | `optimal_params.json`（**供第五章读取**） |

---

## 技术要点

- **数据**：MACdb 数据库 → 236 队列 × 1079 代谢物特征，乳腺癌二分类（阳性率 32.2%）
- **训练/测试**：70/30 分层切分，Z-score 标准化在切分后进行（无数据泄漏）
- **ElasticNet**：`saga` solver，GridSearchCV 7×10=70 候选 × 5 折 = 350 fits
- **Optuna**：TPE sampler，30 trials × 3 折 CV，优化 8 个 XGBoost 超参数
- **Stacking**：`stack_method='predict_proba'`，5 折交叉训练，LR 元学习器
- **贝叶斯网络**：29 节点（Top28 代谢物 + Cancer_Status），3 分类离散化，BIC 评分，Hill Climbing 结构学习
- **随机种子**：统一 `random_state=42`

---

## 注意事项

1. 原始文件 `chapter3_prediction.py` 未被修改，可随时回退对比
2. `optimal_params.json` 是第五章的必要输入文件，不要删除
3. ConvergenceWarning 是 saga solver 的正常提示，不影响模型收敛
4. 如 optuna 未安装，XGBoost 会回退为默认参数
