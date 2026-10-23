"""
HMDB代谢物参考值查询与比对工具
功能：
1. 从HMDB查询代谢物的正常参考浓度范围
2. 与实验数据进行比对
3. 生成类似体检报告的解读
4. 支持批量处理和离线缓存

使用方法：
    python hmdb_reference_query.py --input downloads.metabolite.txt --output reference_report.csv
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HMDBReferenceQuery:
    """
    HMDB代谢物参考值查询器
    """
    
    def __init__(self, cache_dir='hmdb_cache'):
        """
        初始化查询器
        
        Parameters:
        -----------
        cache_dir : str
            缓存目录路径，用于存储已查询的HMDB数据
        """
        self.cache_dir = cache_dir
        self.base_url = "https://hmdb.ca/unearth/q"
        self.api_url = "https://hmdb.ca/metabolites/"
        
        # 创建缓存目录
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"[OK] 创建缓存目录: {cache_dir}")
        
        # 加载缓存
        self.cache_file = os.path.join(cache_dir, 'hmdb_cache.json')
        self.cache = self._load_cache()
        
        print("[查询] HMDB参考值查询器初始化完成")
        print(f"[保存] 缓存文件: {self.cache_file}")
    
    def _load_cache(self):
        """加载本地缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                print(f"[文件] 已加载缓存，包含 {len(cache)} 个代谢物")
                return cache
            except:
                print("[!] 缓存文件损坏，创建新缓存")
                return {}
        return {}
    
    def _save_cache(self):
        """保存缓存到本地"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def query_hmdb_web(self, metabolite_name):
        """
        通过HMDB网站搜索代谢物
        
        Parameters:
        -----------
        metabolite_name : str
            代谢物名称（如 'Creatinine', 'Glucose'）
            
        Returns:
        --------
        dict : 包含参考值信息的字典，如果查询失败返回None
        """
        # 检查缓存
        if metabolite_name in self.cache:
            print(f"  [缓存] 从缓存获取: {metabolite_name}")
            return self.cache[metabolite_name]
        
        try:
            print(f"  [网络] 正在查询HMDB: {metabolite_name}...", end=' ')
            
            # HMDB搜索API
            search_url = f"{self.base_url}/api/search"
            params = {
                'query': metabolite_name,
                'searcher': 'metabolites',
                'type': 'compound'
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('results') and len(data['results']) > 0:
                    # 获取第一个匹配的代谢物
                    first_result = data['results'][0]
                    hmdb_id = first_result.get('identifier', '')
                    name = first_result.get('name', metabolite_name)
                    
                    # 查询详细信息
                    detail_data = self._get_metabolite_detail(hmdb_id)
                    
                    if detail_data:
                        # 提取参考浓度信息
                        ref_data = self._extract_reference_concentrations(
                            detail_data, metabolite_name
                        )
                        
                        # 保存到缓存
                        self.cache[metabolite_name] = ref_data
                        self._save_cache()
                        
                        print("成功")
                        return ref_data
            
            print("未找到")
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"网络错误: {e}")
            return None
        except Exception as e:
            print(f"错误: {e}")
            return None
    
    def _get_metabolite_detail(self, hmdb_id):
        """获取代谢物详细信息"""
        try:
            # 尝试多种格式获取数据
            urls = [
                f"{self.api_url}{hmdb_id}.json",
                f"https://hmdb.ca/metabolites/{hmdb_id}.json",
                f"https://hmdb.ca/metabolites/{hmdb_id}"
            ]
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        if 'json' in url:
                            return response.json()
                        else:
                            # 解析HTML获取浓度信息
                            return self._parse_hmdb_html(response.text, hmdb_id)
                except:
                    continue
            
            return None
        except:
            return None
    
    def _parse_hmdb_html(self, html_content, hmdb_id):
        """从HMDB网页HTML解析浓度信息"""
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 查找Normal Concentrations部分
            data = {
                'hmdb_id': hmdb_id,
                'normal_concentrations': [],
                'abnormal_concentrations': [],
                'name': ''
            }
            
            # 提取名称
            name_tag = soup.find('h1', class_='metabolite-name')
            if name_tag:
                data['name'] = name_tag.text.strip()
            
            # 查找浓度表格
            conc_sections = soup.find_all('div', class_='concentration-section')
            for section in conc_sections:
                biofluid = section.find('h4')
                if biofluid:
                    biofluid_name = biofluid.text.strip()
                    
                    # 提取数值
                    conc_value = section.find('span', class_='concentration-value')
                    if conc_value:
                        value_text = conc_value.text.strip()
                        data['normal_concentrations'].append({
                            'biofluid': biofluid_name,
                            'value': value_text
                        })
            
            return data
        except:
            return None
    
    def _extract_reference_concentrations(self, detail_data, query_name):
        """从详细数据中提取参考浓度"""
        ref_data = {
            'query_name': query_name,
            'hmdb_id': detail_data.get('hmdb_id', 'N/A'),
            'hmdb_name': detail_data.get('name', query_name),
            'biospecimen_locations': [],
            'normal_concentrations': {},
            'abnormal_concentrations': {},
            'query_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'success'
        }
        
        try:
            # 提取生物样本位置
            if 'biospecimen_locations' in detail_data:
                ref_data['biospecimen_locations'] = detail_data['biospecimen_locations']
            
            # 提取正常浓度
            if 'normal_concentrations' in detail_data:
                for conc in detail_data['normal_concentrations']:
                    biofluid = conc.get('biospecimen', 'Unknown')
                    value = conc.get('concentration_value', 'N/A')
                    unit = conc.get('concentration_units', '')
                    
                    if biofluid not in ref_data['normal_concentrations']:
                        ref_data['normal_concentrations'][biofluid] = []
                    
                    ref_data['normal_concentrations'][biofluid].append({
                        'value': value,
                        'unit': unit,
                        'subject_condition': conc.get('subject_condition', 'Normal'),
                        'age': conc.get('patient_age', 'N/A'),
                        'sex': conc.get('patient_sex', 'N/A')
                    })
            
            # 提取异常浓度
            if 'abnormal_concentrations' in detail_data:
                for conc in detail_data['abnormal_concentrations']:
                    biofluid = conc.get('biospecimen', 'Unknown')
                    condition = conc.get('disease', 'Abnormal')
                    
                    if biofluid not in ref_data['abnormal_concentrations']:
                        ref_data['abnormal_concentrations'][biofluid] = []
                    
                    ref_data['abnormal_concentrations'][biofluid].append({
                        'condition': condition,
                        'value': conc.get('concentration_value', 'N/A'),
                        'unit': conc.get('concentration_units', '')
                    })
            
        except Exception as e:
            ref_data['status'] = f'error: {str(e)}'
        
        return ref_data
    
    def get_reference_for_metabolite(self, metabolite_name, tissue_type='Serum'):
        """
        获取单个代谢物的参考值
        
        Parameters:
        -----------
        metabolite_name : str
            代谢物名称
        tissue_type : str
            组织类型（Serum/Plasma/Urine等）
            
        Returns:
        --------
        dict : 参考值信息
        """
        # 标准化代谢物名称
        clean_name = self._clean_metabolite_name(metabolite_name)
        
        # 查询HMDB
        hmdb_data = self.query_hmdb_web(clean_name)
        
        if hmdb_data and hmdb_data.get('status') == 'success':
            # 提取特定组织的参考值
            normal_concs = hmdb_data.get('normal_concentrations', {})
            
            # 尝试匹配组织类型
            tissue_concs = None
            for biofluid, concs in normal_concs.items():
                if tissue_type.lower() in biofluid.lower():
                    tissue_concs = concs
                    break
            
            # 如果没找到特定组织，使用第一个可用的
            if not tissue_concs and normal_concs:
                tissue_concs = list(normal_concs.values())[0]
            
            if tissue_concs:
                # 计算平均参考范围
                values = []
                for conc in tissue_concs:
                    val = conc.get('value', '')
                    if val and val != 'N/A':
                        try:
                            # 提取数值（处理范围如 "10-20"）
                            if '-' in str(val):
                                parts = str(val).split('-')
                                values.append((float(parts[0]) + float(parts[1])) / 2)
                            else:
                                values.append(float(val))
                        except:
                            pass
                
                if values:
                    return {
                        'metabolite': metabolite_name,
                        'hmdb_name': hmdb_data.get('hmdb_name', clean_name),
                        'hmdb_id': hmdb_data.get('hmdb_id', 'N/A'),
                        'reference_mean': np.mean(values),
                        'reference_std': np.std(values),
                        'reference_range': f"{np.min(values):.2f} - {np.max(values):.2f}",
                        'unit': tissue_concs[0].get('unit', 'unknown'),
                        'tissue_type': tissue_type,
                        'data_source': 'HMDB',
                        'query_status': 'success'
                    }
        
        # 查询失败或没有数据
        return {
            'metabolite': metabolite_name,
            'hmdb_name': clean_name,
            'hmdb_id': 'N/A',
            'reference_mean': None,
            'reference_std': None,
            'reference_range': None,
            'unit': None,
            'tissue_type': tissue_type,
            'data_source': 'HMDB',
            'query_status': 'not_found'
        }
    
    def _clean_metabolite_name(self, name):
        """清理代谢物名称，移除脂质编号等修饰符"""
        import re
        
        # 移除常见的修饰符（保留基本名称用于查询）
        clean = name.strip()
        
        # 如果是脂质类（如 PC(38:1)），只保留前缀
        lipid_patterns = [
            r'^(PC|PE|PS|PI|PG|PA|SM|Cer|DG|TG|MG|CL|LPC|LPE)[\s\(\[a-zA-Z0-9:\)]+',
            r'^(HETE|HODE|EET|DiHETE)[\s\(\[a-zA-Z0-9:\)]+'
        ]
        
        for pattern in lipid_patterns:
            match = re.match(pattern, clean)
            if match:
                # 返回脂质类别（如 PC, SM）
                return match.group(1)
        
        # 移除常见的修饰符（如 (OH), C16:1 等）
        clean = re.sub(r'\s*\(.*?\)', '', clean)  # 移除括号内容
        clean = re.sub(r'\s+C\d+:\d+', '', clean)  # 移除碳链标记
        clean = re.sub(r'\s+\w\d+:\d+', '', clean)  # 移除其他标记
        
        return clean.strip()
    
    def batch_query(self, metabolite_list, tissue_mapping=None, delay=1):
        """
        批量查询代谢物参考值
        
        Parameters:
        -----------
        metabolite_list : list
            代谢物名称列表
        tissue_mapping : dict
            代谢物到组织类型的映射（如 {'Creatinine': 'Serum', 'Glucose': 'Plasma'}）
        delay : float
            查询间隔（秒），避免请求过快
            
        Returns:
        --------
        DataFrame : 包含所有代谢物参考值的表格
        """
        results = []
        total = len(metabolite_list)
        
        print(f"\n>> 开始批量查询 {total} 个代谢物...")
        print("=" * 60)
        
        for i, metabolite in enumerate(metabolite_list, 1):
            # 确定组织类型
            tissue = 'Serum'  # 默认
            if tissue_mapping and metabolite in tissue_mapping:
                tissue = tissue_mapping[metabolite]
            
            print(f"\n[{i}/{total}] 查询代谢物: {metabolite}")
            
            result = self.get_reference_for_metabolite(metabolite, tissue)
            results.append(result)
            
            # 延迟，避免请求过快
            if i < total:
                time.sleep(delay)
        
        print("\n" + "=" * 60)
        print(f"[OK] 批量查询完成！成功: {sum(1 for r in results if r['query_status'] == 'success')}/{total}")
        
        return pd.DataFrame(results)
    
    def compare_with_experimental_data(self, experimental_df, metabolite_col='original_metabolite_name',
                                      value_col='case_concentration', tissue_col='Tissue',
                                      cohort_col='Cohort_id'):
        """
        将实验数据与HMDB参考值进行对比
        
        Parameters:
        -----------
        experimental_df : DataFrame
            实验数据表（来自downloads.metabolite.txt）
        metabolite_col : str
            代谢物名称列
        value_col : str
            病例组浓度列
        tissue_col : str
            组织类型列
        cohort_col : str
            队列ID列
            
        Returns:
        --------
        DataFrame : 包含对比结果的报告
        """
        print("\n[统计] 开始比对实验数据与HMDB参考值...")
        
        # 获取唯一代谢物列表
        unique_metabolites = experimental_df[metabolite_col].unique()
        print(f"发现 {len(unique_metabolites)} 个唯一代谢物")
        
        # 查询HMDB参考值
        tissue_mapping = {}
        if tissue_col in experimental_df.columns:
            for _, row in experimental_df.iterrows():
                if pd.notna(row[tissue_col]):
                    tissue_mapping[row[metabolite_col]] = row[tissue_col]
        
        ref_df = self.batch_query(list(unique_metabolites), tissue_mapping)
        
        # 创建参考值字典
        ref_dict = {}
        for _, row in ref_df.iterrows():
            ref_dict[row['metabolite']] = row
        
        # 比对每条实验数据
        comparison_results = []
        
        for _, exp_row in experimental_df.iterrows():
            metabolite = exp_row[metabolite_col]
            measured_value = exp_row[value_col]
            control_value = exp_row.get('control_concentration', None)
            log2fc = exp_row.get('log2FC', None)
            p_value = exp_row.get('case_control_p-value', None)
            
            ref_info = ref_dict.get(metabolite, {})
            
            # 生成比对结果
            comparison = self._generate_comparison_report(
                metabolite, measured_value, control_value, 
                log2fc, p_value, ref_info
            )
            
            comparison['cohort_id'] = exp_row.get(cohort_col, 'N/A')
            comparison['tissue'] = exp_row.get(tissue_col, 'Unknown')
            
            comparison_results.append(comparison)
        
        return pd.DataFrame(comparison_results)
    
    def _generate_comparison_report(self, metabolite, measured_value, control_value,
                                   log2fc, p_value, ref_info):
        """生成单个代谢物的比对报告（类似体检报告格式）"""
        report = {
            'metabolite': metabolite,
            'measured_concentration': measured_value,
            'control_concentration': control_value,
            'log2_fold_change': log2fc,
            'p_value': p_value,
            'hmdb_reference_mean': ref_info.get('reference_mean'),
            'hmdb_reference_range': ref_info.get('reference_range'),
            'hmdb_unit': ref_info.get('unit'),
            'hmdb_status': ref_info.get('query_status', 'unknown'),
            'interpretation': '',
            'clinical_significance': '',
            'abnormal_flag': ''
        }
        
        # 检查是否有HMDB参考值
        if ref_info.get('query_status') != 'success' or not ref_info.get('reference_mean'):
            report['interpretation'] = '[!] HMDB无此代谢物参考值'
            report['clinical_significance'] = '需参考对照组数据判定'
            report['abnormal_flag'] = 'N/A'
            return report
        
        ref_mean = ref_info['reference_mean']
        ref_range = ref_info.get('reference_range', '')
        
        # 判定是否异常（基于对照组或HMDB参考值）
        reference_to_use = ref_mean  # 优先使用HMDB参考值
        
        if pd.notna(measured_value) and reference_to_use:
            # 计算偏离程度
            deviation = (measured_value - reference_to_use) / reference_to_use * 100
            
            # 设定阈值（类似体检报告）
            if abs(deviation) < 10:
                report['abnormal_flag'] = '正常'
                report['interpretation'] = f'在正常范围内（偏离{deviation:+.1f}%）'
            elif abs(deviation) < 25:
                report['abnormal_flag'] = '边缘升高' if deviation > 0 else '边缘降低'
                report['interpretation'] = f'轻度偏离（偏离{deviation:+.1f}%）'
            else:
                report['abnormal_flag'] = '明显升高' if deviation > 0 else '明显降低'
                report['interpretation'] = f'显著偏离（偏离{deviation:+.1f}%）'
            
            # 临床意义判定
            if pd.notna(log2fc) and pd.notna(p_value):
                if p_value < 0.05:
                    if log2fc > 1:
                        report['clinical_significance'] = '[!] 显著升高，可能与疾病相关'
                    elif log2fc < -1:
                        report['clinical_significance'] = '[!] 显著降低，可能与疾病相关'
                    else:
                        report['clinical_significance'] = '有统计学差异，但幅度中等'
                else:
                    report['clinical_significance'] = '无显著差异（p>=0.05）'
            else:
                report['clinical_significance'] = '缺乏统计学检验数据'
        else:
            report['interpretation'] = '数据缺失，无法判定'
            report['abnormal_flag'] = 'N/A'
        
        return report


def main():
    """
    主程序：示例用法
    """
    print("=" * 80)
    print("HMDB代谢物参考值查询与比对工具")
    print("=" * 80)
    
    # 初始化查询器
    query_tool = HMDBReferenceQuery(cache_dir='hmdb_cache')
    
    # 示例1：查询单个代谢物
    print("\n" + "=" * 80)
    print("示例1：查询单个代谢物")
    print("=" * 80)
    
    test_metabolites = ['Creatinine', 'Glucose', 'Cholesterol', 'Urea']
    
    for metabolite in test_metabolites:
        result = query_tool.get_reference_for_metabolite(metabolite, 'Serum')
        
        print(f"\n[报告] {metabolite}:")
        print(f"   HMDB名称: {result['hmdb_name']}")
        print(f"   HMDB ID: {result['hmdb_id']}")
        if result['query_status'] == 'success':
            print(f"   参考范围: {result['reference_range']} {result['unit']}")
            print(f"   平均值: {result['reference_mean']:.2f}")
        else:
            print(f"   状态: 未找到HMDB参考值")
    
    # 示例2：批量查询
    print("\n" + "=" * 80)
    print("示例2：批量查询（演示）")
    print("=" * 80)
    
    metabolite_list = [
        'Creatinine', 'Glucose', 'Lactate', 'Pyruvate',
        'Cholesterol', 'Bilirubin', 'Uric acid'
    ]
    
    ref_df = query_tool.batch_query(metabolite_list, delay=2)
    
    print("\n批量查询结果预览：")
    print(ref_df[['metabolite', 'hmdb_name', 'reference_range', 'unit', 'query_status']].to_string(index=False))
    
    # 保存结果
    output_file = 'hmdb_reference_values.csv'
    ref_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n[保存] 参考值已保存至: {output_file}")
    
    # 示例3：与实验数据比对
    print("\n" + "=" * 80)
    print("示例3：与实验数据比对（演示）")
    print("=" * 80)
    
    # 创建示例实验数据
    sample_data = pd.DataFrame({
        'original_metabolite_name': ['Creatinine', 'Glucose', 'Cholesterol'],
        'case_concentration': [0.82, 97.0, 180.0],
        'control_concentration': [0.83, 96.0, 175.0],
        'log2FC': [-0.02, 0.01, 0.04],
        'case_control_p-value': [0.324, 0.197, 0.450],
        'Tissue': ['Serum', 'Serum', 'Serum'],
        'Cohort_id': ['METAC_1', 'METAC_1', 'METAC_1']
    })
    
    comparison_df = query_tool.compare_with_experimental_data(sample_data)
    
    print("\n比对结果预览：")
    print(comparison_df[['metabolite', 'measured_concentration', 'hmdb_reference_range', 
                        'abnormal_flag', 'interpretation']].to_string(index=False))
    
    # 保存比对报告
    report_file = 'metabolite_comparison_report.csv'
    comparison_df.to_csv(report_file, index=False, encoding='utf-8-sig')
    print(f"\n[保存] 比对报告已保存至: {report_file}")
    
    print("\n" + "=" * 80)
    print("[OK] 演示完成！")
    print("=" * 80)
    print("\n[提示] 使用提示：")
    print("   1. 批量查询时建议设置 delay=1-2 秒，避免请求过快")
    print("   2. 查询结果会自动缓存，下次查询更快")
    print("   3. 脂质类代谢物（如PC、SM）可能在HMDB中找不到具体编号的标准")
    print("   4. 建议使用对照组数据作为主要参考标准")
    print("\n[实验] 实际使用您的数据：")
    print("   python hmdb_reference_query.py --input downloads.metabolite.txt --study downloads.study.txt")


if __name__ == '__main__':
    main()
