#!/usr/bin/env python3
"""
PIPL隐私保护LLM框架 - 完整功能测试

本脚本提供全面的功能测试，验证PIPL隐私保护LLM框架的所有核心功能：
- PII检测功能测试
- 差分隐私保护测试
- 合规性监控测试
- 端到端工作流程测试
- 性能基准测试
- 错误处理测试

使用方法:
1. 在Colab环境中运行此脚本
2. 确保已加载Qwen2.5-7B模型和PIPL集成代码
3. 查看详细的测试报告

作者: PIPL Framework Team
版本: 1.0.0
日期: 2025-10-23
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import traceback

class ComprehensiveFunctionalTest:
    """完整功能测试类"""
    
    def __init__(self):
        self.test_results = {}
        self.test_cases = []
        self.performance_metrics = {}
        self.error_logs = []
        
    def print_test_header(self, test_name: str):
        """打印测试标题"""
        print("\n" + "="*80)
        print(f"🧪 {test_name}")
        print("="*80)
    
    def print_test_result(self, test_name: str, success: bool, details: str = ""):
        """打印测试结果"""
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} {test_name}")
        if details:
            print(f"   详情: {details}")
    
    def test_pii_detection(self, privacy_qwen):
        """测试PII检测功能"""
        self.print_test_header("PII检测功能测试")
        
        test_cases = [
            {
                'text': '用户张三，电话13812345678，邮箱zhangsan@example.com',
                'expected_pii': ['name', 'phone', 'email'],
                'description': '包含多种PII信息'
            },
            {
                'text': '身份证号码：110101199001011234',
                'expected_pii': ['id_card'],
                'description': '身份证号码检测'
            },
            {
                'text': '这个产品很不错，我很满意。',
                'expected_pii': [],
                'description': '无PII信息'
            },
            {
                'text': '李四觉得这个服务很糟糕，完全不推荐。',
                'expected_pii': ['name'],
                'description': '中文姓名检测'
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            try:
                # 使用PII检测器
                pii_result = privacy_qwen.pii_detector.detect(case['text'])
                detected_types = [pii['type'] for pii in pii_result]
                
                # 验证检测结果
                success = set(detected_types) == set(case['expected_pii'])
                
                self.print_test_result(
                    f"测试案例 {i+1}: {case['description']}",
                    success,
                    f"检测到: {detected_types}, 期望: {case['expected_pii']}"
                )
                
                results.append({
                    'case': case,
                    'detected': detected_types,
                    'expected': case['expected_pii'],
                    'success': success,
                    'pii_count': len(pii_result)
                })
                
            except Exception as e:
                self.print_test_result(f"测试案例 {i+1}: {case['description']}", False, f"错误: {e}")
                results.append({
                    'case': case,
                    'error': str(e),
                    'success': False
                })
        
        # 统计结果
        successful = len([r for r in results if r.get('success', False)])
        total = len(results)
        
        self.test_results['pii_detection'] = {
            'total_tests': total,
            'successful_tests': successful,
            'success_rate': successful / total * 100,
            'results': results
        }
        
        print(f"\n📊 PII检测测试总结: {successful}/{total} 通过 ({successful/total*100:.1f}%)")
        return results
    
    def test_differential_privacy(self, privacy_qwen):
        """测试差分隐私功能"""
        self.print_test_header("差分隐私功能测试")
        
        test_cases = [
            {
                'data': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                'epsilon': 1.0,
                'description': '基础差分隐私测试'
            },
            {
                'data': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                'epsilon': 0.5,
                'description': '高隐私保护测试'
            },
            {
                'data': np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
                'epsilon': 2.0,
                'description': '低隐私保护测试'
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            try:
                # 测试差分隐私
                original_data = case['data'].copy()
                noisy_data = privacy_qwen.differential_privacy.add_noise(original_data, case['epsilon'])
                
                # 验证噪声添加
                noise_added = not np.array_equal(original_data, noisy_data)
                noise_magnitude = np.linalg.norm(noisy_data - original_data)
                
                success = noise_added and noise_magnitude > 0
                
                self.print_test_result(
                    f"测试案例 {i+1}: {case['description']}",
                    success,
                    f"噪声幅度: {noise_magnitude:.4f}, Epsilon: {case['epsilon']}"
                )
                
                results.append({
                    'case': case,
                    'original_data': original_data,
                    'noisy_data': noisy_data,
                    'noise_magnitude': noise_magnitude,
                    'success': success
                })
                
            except Exception as e:
                self.print_test_result(f"测试案例 {i+1}: {case['description']}", False, f"错误: {e}")
                results.append({
                    'case': case,
                    'error': str(e),
                    'success': False
                })
        
        # 统计结果
        successful = len([r for r in results if r.get('success', False)])
        total = len(results)
        
        self.test_results['differential_privacy'] = {
            'total_tests': total,
            'successful_tests': successful,
            'success_rate': successful / total * 100,
            'results': results
        }
        
        print(f"\n📊 差分隐私测试总结: {successful}/{total} 通过 ({successful/total*100:.1f}%)")
        return results
    
    def test_compliance_monitoring(self, privacy_qwen):
        """测试合规性监控功能"""
        self.print_test_header("合规性监控功能测试")
        
        test_cases = [
            {
                'data': {'type': 'personal_info', 'risk_level': 'low', 'cross_border': False},
                'expected_status': 'compliant',
                'description': '低风险合规测试'
            },
            {
                'data': {'type': 'sensitive_info', 'risk_level': 'high', 'cross_border': False},
                'expected_status': 'non_compliant',
                'description': '高风险合规测试'
            },
            {
                'data': {'type': 'general', 'risk_level': 'medium', 'cross_border': True},
                'expected_status': 'compliant',
                'description': '跨境传输测试'
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            try:
                # 测试合规性检查
                compliance = privacy_qwen.compliance_monitor.check_compliance(case['data'])
                
                # 验证合规状态
                success = compliance['status'] == case['expected_status']
                
                self.print_test_result(
                    f"测试案例 {i+1}: {case['description']}",
                    success,
                    f"状态: {compliance['status']}, 期望: {case['expected_status']}"
                )
                
                results.append({
                    'case': case,
                    'compliance': compliance,
                    'success': success
                })
                
            except Exception as e:
                self.print_test_result(f"测试案例 {i+1}: {case['description']}", False, f"错误: {e}")
                results.append({
                    'case': case,
                    'error': str(e),
                    'success': False
                })
        
        # 测试操作记录
        try:
            operation = {
                'operation_id': 'test_operation',
                'operation_type': 'compliance_test',
                'user_id': 'test_user',
                'data_type': 'test_data'
            }
            log_success = privacy_qwen.compliance_monitor.log_operation(operation)
            
            self.print_test_result("操作记录测试", log_success, "操作记录功能正常")
            
        except Exception as e:
            self.print_test_result("操作记录测试", False, f"错误: {e}")
        
        # 统计结果
        successful = len([r for r in results if r.get('success', False)])
        total = len(results)
        
        self.test_results['compliance_monitoring'] = {
            'total_tests': total,
            'successful_tests': successful,
            'success_rate': successful / total * 100,
            'results': results
        }
        
        print(f"\n📊 合规性监控测试总结: {successful}/{total} 通过 ({successful/total*100:.1f}%)")
        return results
    
    def test_end_to_end_workflow(self, privacy_qwen):
        """测试端到端工作流程"""
        self.print_test_header("端到端工作流程测试")
        
        test_cases = [
            {
                'text': '请介绍一下人工智能的发展历史。',
                'expected_risk': 'low',
                'description': '普通文本处理'
            },
            {
                'text': '用户张三，电话13812345678，对这个产品很满意。',
                'expected_risk': 'high',
                'description': '包含PII的文本处理'
            },
            {
                'text': '李四觉得这个服务很糟糕，完全不推荐。',
                'expected_risk': 'medium',
                'description': '包含姓名的文本处理'
            },
            {
                'text': '整体来说比较满意，会继续使用。',
                'expected_risk': 'low',
                'description': '无敏感信息文本处理'
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            try:
                # 执行端到端处理
                result = privacy_qwen.generate_with_privacy_protection(case['text'])
                
                # 验证处理结果
                success = (
                    result.get('status') == 'success' and
                    result.get('risk_level') == case['expected_risk'] and
                    'response' in result and
                    'compliance' in result
                )
                
                self.print_test_result(
                    f"测试案例 {i+1}: {case['description']}",
                    success,
                    f"风险级别: {result.get('risk_level')}, 响应长度: {len(result.get('response', ''))}"
                )
                
                results.append({
                    'case': case,
                    'result': result,
                    'success': success
                })
                
            except Exception as e:
                self.print_test_result(f"测试案例 {i+1}: {case['description']}", False, f"错误: {e}")
                results.append({
                    'case': case,
                    'error': str(e),
                    'success': False
                })
        
        # 统计结果
        successful = len([r for r in results if r.get('success', False)])
        total = len(results)
        
        self.test_results['end_to_end_workflow'] = {
            'total_tests': total,
            'successful_tests': successful,
            'success_rate': successful / total * 100,
            'results': results
        }
        
        print(f"\n📊 端到端工作流程测试总结: {successful}/{total} 通过 ({successful/total*100:.1f}%)")
        return results
    
    def test_performance_benchmark(self, privacy_qwen):
        """测试性能基准"""
        self.print_test_header("性能基准测试")
        
        test_cases = [
            {'text': '请介绍一下机器学习。', 'iterations': 5},
            {'text': '用户张三，电话13812345678，对这个产品很满意。', 'iterations': 3},
            {'text': '这个服务很不错，推荐使用。', 'iterations': 5}
        ]
        
        performance_results = []
        
        for i, case in enumerate(test_cases):
            try:
                print(f"执行性能测试 {i+1}: {case['text'][:20]}...")
                
                times = []
                for j in range(case['iterations']):
                    start_time = time.time()
                    result = privacy_qwen.generate_with_privacy_protection(case['text'])
                    end_time = time.time()
                    
                    if result.get('status') == 'success':
                        times.append(end_time - start_time)
                
                if times:
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    min_time = np.min(times)
                    max_time = np.max(times)
                    
                    self.print_test_result(
                        f"性能测试 {i+1}",
                        True,
                        f"平均时间: {avg_time:.3f}s, 标准差: {std_time:.3f}s"
                    )
                    
                    performance_results.append({
                        'case': case,
                        'avg_time': avg_time,
                        'std_time': std_time,
                        'min_time': min_time,
                        'max_time': max_time,
                        'iterations': len(times)
                    })
                else:
                    self.print_test_result(f"性能测试 {i+1}", False, "所有迭代都失败")
                    
            except Exception as e:
                self.print_test_result(f"性能测试 {i+1}", False, f"错误: {e}")
        
        # 记录性能指标
        self.performance_metrics = {
            'test_cases': len(test_cases),
            'results': performance_results
        }
        
        print(f"\n📊 性能基准测试完成: {len(performance_results)} 个测试案例")
        return performance_results
    
    def test_error_handling(self, privacy_qwen):
        """测试错误处理"""
        self.print_test_header("错误处理测试")
        
        test_cases = [
            {
                'input': None,
                'expected_error': True,
                'description': '空输入测试'
            },
            {
                'input': '',
                'expected_error': False,
                'description': '空字符串测试'
            },
            {
                'input': 'a' * 10000,
                'expected_error': False,
                'description': '超长文本测试'
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            try:
                if case['input'] is None:
                    # 测试空输入
                    try:
                        result = privacy_qwen.generate_with_privacy_protection(None)
                        error_handled = result.get('status') == 'failed'
                    except:
                        error_handled = True
                else:
                    # 测试正常输入
                    result = privacy_qwen.generate_with_privacy_protection(case['input'])
                    error_handled = result.get('status') != 'failed'
                
                success = error_handled == case['expected_error']
                
                self.print_test_result(
                    f"测试案例 {i+1}: {case['description']}",
                    success,
                    f"错误处理: {'正常' if error_handled else '异常'}"
                )
                
                results.append({
                    'case': case,
                    'error_handled': error_handled,
                    'success': success
                })
                
            except Exception as e:
                self.print_test_result(f"测试案例 {i+1}: {case['description']}", False, f"错误: {e}")
                results.append({
                    'case': case,
                    'error': str(e),
                    'success': False
                })
        
        # 统计结果
        successful = len([r for r in results if r.get('success', False)])
        total = len(results)
        
        self.test_results['error_handling'] = {
            'total_tests': total,
            'successful_tests': successful,
            'success_rate': successful / total * 100,
            'results': results
        }
        
        print(f"\n📊 错误处理测试总结: {successful}/{total} 通过 ({successful/total*100:.1f}%)")
        return results
    
    def test_batch_processing(self, privacy_qwen):
        """测试批量处理"""
        self.print_test_header("批量处理测试")
        
        batch_texts = [
            "这个产品很不错。",
            "张三觉得服务很差。",
            "整体比较满意。",
            "用户李四，电话13987654321，对这个产品很满意。",
            "这个服务值得推荐。"
        ]
        
        try:
            # 执行批量处理
            results = []
            for i, text in enumerate(batch_texts):
                result = privacy_qwen.generate_with_privacy_protection(text)
                result['batch_index'] = i
                results.append(result)
            
            # 统计结果
            successful = len([r for r in results if r.get('status') == 'success'])
            total = len(results)
            
            self.print_test_result(
                "批量处理测试",
                successful == total,
                f"成功: {successful}/{total}"
            )
            
            # 显示批量处理结果
            print("\n📋 批量处理结果:")
            for i, result in enumerate(results):
                status = "✅" if result.get('status') == 'success' else "❌"
                risk = result.get('risk_level', 'unknown')
                print(f"  {status} 案例 {i+1}: 风险级别 {risk}")
            
            self.test_results['batch_processing'] = {
                'total_tests': total,
                'successful_tests': successful,
                'success_rate': successful / total * 100,
                'results': results
            }
            
            return results
            
        except Exception as e:
            self.print_test_result("批量处理测试", False, f"错误: {e}")
            return []
    
    def generate_comprehensive_report(self):
        """生成综合测试报告"""
        self.print_test_header("综合测试报告")
        
        # 计算总体统计
        total_tests = sum(result['total_tests'] for result in self.test_results.values())
        total_successful = sum(result['successful_tests'] for result in self.test_results.values())
        overall_success_rate = (total_successful / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 总体测试统计:")
        print(f"  总测试数: {total_tests}")
        print(f"  成功测试: {total_successful}")
        print(f"  成功率: {overall_success_rate:.1f}%")
        
        print(f"\n📋 各模块测试结果:")
        for module, result in self.test_results.items():
            print(f"  {module}: {result['successful_tests']}/{result['total_tests']} ({result['success_rate']:.1f}%)")
        
        # 性能统计
        if self.performance_metrics:
            print(f"\n⚡ 性能统计:")
            for result in self.performance_metrics['results']:
                print(f"  平均响应时间: {result['avg_time']:.3f}s")
                print(f"  响应时间标准差: {result['std_time']:.3f}s")
        
        # 生成详细报告
        report = {
            'test_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_statistics': {
                'total_tests': total_tests,
                'successful_tests': total_successful,
                'success_rate': overall_success_rate
            },
            'module_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'test_summary': {
                'pii_detection': self.test_results.get('pii_detection', {}).get('success_rate', 0),
                'differential_privacy': self.test_results.get('differential_privacy', {}).get('success_rate', 0),
                'compliance_monitoring': self.test_results.get('compliance_monitoring', {}).get('success_rate', 0),
                'end_to_end_workflow': self.test_results.get('end_to_end_workflow', {}).get('success_rate', 0),
                'error_handling': self.test_results.get('error_handling', {}).get('success_rate', 0),
                'batch_processing': self.test_results.get('batch_processing', {}).get('success_rate', 0)
            }
        }
        
        # 保存报告
        report_path = 'comprehensive_test_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细测试报告已保存: {report_path}")
        
        return report
    
    def run_all_tests(self, privacy_qwen):
        """运行所有测试"""
        print("🚀 开始完整功能测试")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # 1. PII检测测试
            self.test_pii_detection(privacy_qwen)
            
            # 2. 差分隐私测试
            self.test_differential_privacy(privacy_qwen)
            
            # 3. 合规性监控测试
            self.test_compliance_monitoring(privacy_qwen)
            
            # 4. 端到端工作流程测试
            self.test_end_to_end_workflow(privacy_qwen)
            
            # 5. 性能基准测试
            self.test_performance_benchmark(privacy_qwen)
            
            # 6. 错误处理测试
            self.test_error_handling(privacy_qwen)
            
            # 7. 批量处理测试
            self.test_batch_processing(privacy_qwen)
            
            # 8. 生成综合报告
            report = self.generate_comprehensive_report()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n🎉 完整功能测试完成!")
            print(f"⏱️ 总耗时: {total_time:.2f} 秒")
            print(f"📊 总体成功率: {report['overall_statistics']['success_rate']:.1f}%")
            
            return report
            
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            traceback.print_exc()
            return None

def main():
    """主函数"""
    print("PIPL隐私保护LLM框架 - 完整功能测试")
    print("="*80)
    
    # 检查是否已加载模型和集成代码
    try:
        # 假设privacy_qwen已经创建
        if 'privacy_qwen' not in globals():
            print("❌ 请先运行PIPL集成代码创建privacy_qwen对象")
            print("请运行: exec(open('colab_pipl_integration.py').read())")
            return
        
        # 创建测试实例
        test_suite = ComprehensiveFunctionalTest()
        
        # 运行所有测试
        report = test_suite.run_all_tests(privacy_qwen)
        
        if report:
            print("\n🎉 测试完成！请查看详细报告了解测试结果。")
        else:
            print("\n❌ 测试失败，请检查错误信息。")
            
    except Exception as e:
        print(f"❌ 测试初始化失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
