#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMDB代谢物参考值查询工具 - 命令行版本
使用示例：
    python query_hmdb_for_my_data.py --metabolite downloads.metabolite.txt --study downloads.study.txt --output my_reference_report.csv
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os

# 导入主要的查询类
from hmdb_reference_query import HMDBReferenceQuery


def load_and_merge_data(metabolite_file, study_file):
    """
    加载并合并代谢物数据与研究数据
    
    Parameters:
    -----------
    metabolite_file : str
        代谢物数据文件路径（downloads.metabolite.txt）
    study_file : str
        研究数据文件路径（downloads.study.txt）
        
    Returns:
    --------
    DataFrame : 合并后的数据
    """
    print(f"\n[文件] 加载数据文件...")
    
    # 加载代谢物数据
    try:
        metabolite_df = pd.read_csv(metabolite_file, sep='\t')
        print(f"   [OK] 代谢物数据: {len(metabolite_df)} 条记录")
    except Exception as e:
        print(f"   [X] 无法加载代谢物数据: {e}")
        sys.exit(1)
    
    # 加载研究数据
    try:
        study_df = pd.read_csv(study_file, sep='\t')
        print(f"   [OK] 研究数据: {len(study_df)} 条记录")
    except Exception as e:
        print(f"   [!] 无法加载研究数据: {e}")
        study_df = None
    
    # 合并数据（添加组织类型信息）
    if study_df is not None and 'Cohort_id' in metabolite_df.columns:
        # 提取每个队列的组织类型
        cohort_tissue = study_df[['Cohort_id', 'Tissue']].drop_duplicates()
        
        # 合并
        merged_df = metabolite_df.merge(cohort_tissue, on='Cohort_id', how='left')
        print(f"   [OK] 数据合并完成: {len(merged_df)} 条记录")
        return merged_df
    
    return metabolite_df


def extract_unique_metabolites(df):
    """
    提取唯一的代谢物列表
    
    Parameters:
    -----------
    df : DataFrame
        合并后的数据表
        
    Returns:
    --------
    list : 唯一代谢物名称列表
    """
    if 'original_metabolite_name' not in df.columns:
        print("[X] 错误：数据表中未找到 'original_metabolite_name' 列")
        sys.exit(1)
    
    unique_metabolites = df['original_metabolite_name'].dropna().unique()
    print(f"\n[统计] 发现 {len(unique_metabolites)} 个唯一代谢物")
    
    return list(unique_metabolites)


def create_tissue_mapping(df):
    """
    创建代谢物到组织类型的映射
    
    Parameters:
    -----------
    df : DataFrame
        数据表
        
    Returns:
    --------
    dict : 代谢物->组织类型的映射
    """
    tissue_mapping = {}
    
    if 'Tissue' in df.columns:
        for _, row in df.iterrows():
            metabolite = row.get('original_metabolite_name')
            tissue = row.get('Tissue')
            
            if pd.notna(metabolite) and pd.notna(tissue):
                # 标准化组织类型名称
                tissue = str(tissue).strip()
                
                # 映射到HMDB支持的类型
                if 'blood' in tissue.lower() or 'serum' in tissue.lower():
                    tissue_mapping[metabolite] = 'Blood'
                elif 'plasma' in tissue.lower():
                    tissue_mapping[metabolite] = 'Plasma'
                elif 'urine' in tissue.lower():
                    tissue_mapping[metabolite] = 'Urine'
                elif 'tissue' in tissue.lower() or 'biopsy' in tissue.lower():
                    tissue_mapping[metabolite] = 'Tissue'
                else:
                    tissue_mapping[metabolite] = 'Serum'  # 默认
    
    return tissue_mapping


def query_hmdb_with_progress(query_tool, metabolite_list, tissue_mapping, delay=1):
    """
    带进度显示的HMDB查询
    
    Parameters:
    -----------
    query_tool : HMDBReferenceQuery
        查询工具实例
    metabolite_list : list
        代谢物列表
    tissue_mapping : dict
        组织类型映射
    delay : float
        查询间隔
        
    Returns:
    --------
    DataFrame : HMDB参考值结果
    """
    from tqdm import tqdm
    
    results = []
    total = len(metabolite_list)
    
    print(f"\n>> 开始查询 {total} 个代谢物的HMDB参考值...")
    print("=" * 80)
    
    # 如果没有tqdm，使用简单循环
    try:
        iterator = tqdm(enumerate(metabolite_list, 1), total=total, desc="查询进度")
    except:
        iterator = enumerate(metabolite_list, 1)
    
    for i, metabolite in iterator:
        tissue = tissue_mapping.get(metabolite, 'Serum')
        
        result = query_tool.get_reference_for_metabolite(metabolite, tissue)
        results.append(result)
        
        if i % 10 == 0:
            success_count = sum(1 for r in results if r['query_status'] == 'success')
            print(f"\n   进度: {i}/{total} | 成功: {success_count} | 当前: {metabolite}")
        
        if i < total:
            import time
            time.sleep(delay)
    
    print("\n" + "=" * 80)
    success_count = sum(1 for r in results if r['query_status'] == 'success')
    print(f"[OK] 查询完成！成功获取: {success_count}/{total} 个代谢物的参考值")
    
    return pd.DataFrame(results)


def generate_comparison_report(query_tool, experimental_df, hmdb_ref_df):
    """
    生成详细的比对报告
    
    Parameters:
    -----------
    query_tool : HMDBReferenceQuery
        查询工具
    experimental_df : DataFrame
        实验数据
    hmdb_ref_df : DataFrame
        HMDB参考值数据
        
    Returns:
    --------
    DataFrame : 比对报告
    """
    print("\n[统计] 生成比对报告...")
    
    # 创建参考值字典
    ref_dict = {}
    for _, row in hmdb_ref_df.iterrows():
        ref_dict[row['metabolite']] = row.to_dict()
    
    # 为每条实验记录生成报告
    reports = []
    
    for idx, exp_row in experimental_df.iterrows():
        metabolite = exp_row.get('original_metabolite_name', 'Unknown')
        measured = exp_row.get('case_concentration')
        control = exp_row.get('control_concentration')
        log2fc = exp_row.get('log2FC')
        p_value = exp_row.get('case_control_p-value')
        cohort = exp_row.get('Cohort_id', 'N/A')
        tissue = exp_row.get('Tissue', 'Unknown')
        
        ref_info = ref_dict.get(metabolite, {})
        
        # 使用query_tool的方法生成报告
        report = query_tool._generate_comparison_report(
            metabolite, measured, control, log2fc, p_value, ref_info
        )
        
        report['record_id'] = idx
        report['cohort_id'] = cohort
        report['tissue'] = tissue
        
        reports.append(report)
    
    report_df = pd.DataFrame(reports)
    
    print(f"   [OK] 生成 {len(report_df)} 条比对记录")
    return report_df


def print_summary_stats(report_df):
    """
    打印统计摘要
    
    Parameters:
    -----------
    report_df : DataFrame
        比对报告
    """
    print("\n" + "=" * 80)
    print("[图表] 统计摘要")
    print("=" * 80)
    
    # HMDB覆盖情况
    total = len(report_df)
    with_hmdb = report_df[report_df['hmdb_status'] == 'success']
    coverage = len(with_hmdb) / total * 100 if total > 0 else 0
    
    print(f"\n1. HMDB参考值覆盖率: {coverage:.1f}% ({len(with_hmdb)}/{total})")
    
    # 异常标记统计
    if 'abnormal_flag' in report_df.columns:
        flag_counts = report_df['abnormal_flag'].value_counts()
        print(f"\n2. 代谢物状态分布:")
        for flag, count in flag_counts.items():
            print(f"   • {flag}: {count} ({count/total*100:.1f}%)")
    
    # 临床意义统计
    if 'clinical_significance' in report_df.columns:
        sig_counts = report_df['clinical_significance'].value_counts().head(5)
        print(f"\n3. 临床意义分布（Top 5）:")
        for sig, count in sig_counts.items():
            print(f"   • {sig}: {count}")
    
    # 示例异常代谢物
    abnormal_df = report_df[report_df['abnormal_flag'].str.contains('高|低', na=False)]
    if len(abnormal_df) > 0:
        print(f"\n4. 异常代谢物示例（Top 5）:")
        for _, row in abnormal_df.head(5).iterrows():
            print(f"   • {row['metabolite']}: {row['abnormal_flag']} - {row['interpretation']}")


def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(
        description='查询HMDB代谢物参考值并与实验数据比对',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python query_hmdb_for_my_data.py -m downloads.metabolite.txt -s downloads.study.txt
  
  # 指定输出文件
  python query_hmdb_for_my_data.py -m downloads.metabolite.txt -s downloads.study.txt -o my_report.csv
  
  # 调整查询速度（避免请求过快）
  python query_hmdb_for_my_data.py -m downloads.metabolite.txt -d 2
        """
    )
    
    parser.add_argument('-m', '--metabolite', required=True,
                       help='代谢物数据文件路径 (downloads.metabolite.txt)')
    parser.add_argument('-s', '--study', required=True,
                       help='研究数据文件路径 (downloads.study.txt)')
    parser.add_argument('-o', '--output', default='hmdb_comparison_report.csv',
                       help='输出文件路径 (默认: hmdb_comparison_report.csv)')
    parser.add_argument('-d', '--delay', type=float, default=1.0,
                       help='查询间隔（秒），默认为1秒')
    parser.add_argument('--cache-dir', default='hmdb_cache',
                       help='缓存目录 (默认: hmdb_cache)')
    parser.add_argument('--test', action='store_true',
                       help='测试模式：只查询前10个代谢物')
    
    args = parser.parse_args()
    
    # 显示欢迎信息
    print("=" * 80)
    print("HMDB代谢物参考值查询工具")
    print("=" * 80)
    print(f"输入文件: {args.metabolite}")
    print(f"研究数据: {args.study}")
    print(f"输出文件: {args.output}")
    print(f"查询间隔: {args.delay} 秒")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.metabolite):
        print(f"\n[错误] 找不到文件 {args.metabolite}")
        sys.exit(1)
    
    if not os.path.exists(args.study):
        print(f"\n[警告] 找不到文件 {args.study}，将使用默认组织类型")
    
    try:
        # 步骤1: 加载数据
        merged_df = load_and_merge_data(args.metabolite, args.study)
        
        # 步骤2: 提取唯一代谢物
        metabolite_list = extract_unique_metabolites(merged_df)
        
        # 测试模式：只查询前10个
        if args.test:
            print(f"\n[测试] 测试模式: 仅查询前10个代谢物")
            metabolite_list = metabolite_list[:10]
        
        # 步骤3: 创建组织类型映射
        tissue_mapping = create_tissue_mapping(merged_df)
        print(f"   [OK] 组织类型映射: {len(tissue_mapping)} 个代谢物")
        
        # 步骤4: 初始化HMDB查询器
        query_tool = HMDBReferenceQuery(cache_dir=args.cache_dir)
        
        # 步骤5: 查询HMDB参考值
        hmdb_ref_df = query_hmdb_with_progress(
            query_tool, metabolite_list, tissue_mapping, delay=args.delay
        )
        
        # 步骤6: 生成比对报告
        report_df = generate_comparison_report(query_tool, merged_df, hmdb_ref_df)
        
        # 步骤7: 打印统计摘要
        print_summary_stats(report_df)
        
        # 步骤8: 保存结果
        report_df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\n[保存] 完整报告已保存至: {args.output}")
        
        # 同时保存HMDB参考值
        ref_output = args.output.replace('.csv', '_reference_values.csv')
        hmdb_ref_df.to_csv(ref_output, index=False, encoding='utf-8-sig')
        print(f"[保存] HMDB参考值已保存至: {ref_output}")
        
        # 保存成功查询的代谢物列表
        success_metabolites = hmdb_ref_df[hmdb_ref_df['query_status'] == 'success']['metabolite'].tolist()
        with open('hmdb_success_metabolites.txt', 'w', encoding='utf-8') as f:
            for m in success_metabolites:
                f.write(f"{m}\n")
        print(f"[保存] 成功查询的代谢物列表: hmdb_success_metabolites.txt")
        
        print("\n" + "=" * 80)
        print("[OK] 全部完成！")
        print("=" * 80)
        
        print("\n[报告] 报告解读指南:")
        print("   • abnormal_flag: 类似体检报告的 高低 标记")
        print("   • interpretation: 详细解读")
        print("   • clinical_significance: 临床意义说明")
        print("   • hmdb_reference_range: HMDB官方参考范围")
        print("   • hmdb_status: 'success' = 找到参考值, 'not_found' = 未找到")
        
        print("\n[!]  注意事项:")
        print("   1. 并非所有代谢物在HMDB中都有参考值")
        print("   2. 脂质类代谢物（如PC、SM）通常没有HMDB具体编号的标准")
        print("   3. 建议以对照组数据作为主要参考标准")
        print("   4. HMDB数据基于文献汇总，可能与您的检测方法有差异")
        
    except KeyboardInterrupt:
        print("\n\n[!] 用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[X] 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
