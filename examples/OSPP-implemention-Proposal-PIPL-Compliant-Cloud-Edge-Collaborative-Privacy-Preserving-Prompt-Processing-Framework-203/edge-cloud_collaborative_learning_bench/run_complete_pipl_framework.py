#!/usr/bin/env python3
"""
完整的Ianvs PIPL隐私保护云边协同提示处理框架执行脚本

在Colab环境中按顺序执行所有10个阶段，实现完整的PIPL合规隐私保护框架
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompletePIPLFrameworkRunner:
    """完整的PIPL框架运行器"""
    
    def __init__(self, base_path="/content/ianvs_pipl_framework"):
        """初始化运行器"""
        self.base_path = base_path
        self.stages = [
            "01_environment_setup",
            "02_dependencies_installation", 
            "03_ianvs_framework_setup",
            "04_dataset_preparation",
            "05_model_deployment",
            "06_privacy_modules_init",
            "07_collaborative_workflow",
            "08_performance_monitoring",
            "09_storymanager_export",
            "10_results_analysis"
        ]
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # 创建基础目录
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(os.path.join(base_path, "logs"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "results"), exist_ok=True)
        
        logger.info(f"完整PIPL框架运行器初始化完成: {self.base_path}")
    
    def run_stage(self, stage_name: str) -> bool:
        """运行单个阶段"""
        print(f"\n{'='*60}")
        print(f"🚀 执行阶段: {stage_name}")
        print(f"{'='*60}")
        
        try:
            # 构建阶段文件路径
            stage_file = os.path.join(self.base_path, f"{stage_file}.py")
            
            if not os.path.exists(stage_file):
                print(f"❌ 阶段文件不存在: {stage_file}")
                return False
            
            # 执行阶段
            start_time = time.time()
            
            # 使用exec执行阶段文件
            with open(stage_file, 'r', encoding='utf-8') as f:
                stage_code = f.read()
            
            # 创建执行环境
            exec_globals = {
                '__name__': '__main__',
                '__file__': stage_file,
                'os': os,
                'sys': sys,
                'json': json,
                'time': time,
                'datetime': datetime,
                'logging': logging,
                'logger': logger
            }
            
            # 执行阶段代码
            exec(stage_code, exec_globals)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 记录结果
            self.results[stage_name] = {
                "status": "success",
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ 阶段 {stage_name} 执行成功 (耗时: {execution_time:.2f}秒)")
            return True
            
        except Exception as e:
            print(f"❌ 阶段 {stage_name} 执行失败: {e}")
            logger.error(f"阶段 {stage_name} 执行失败: {e}")
            
            self.results[stage_name] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return False
    
    def run_all_stages(self) -> Dict[str, Any]:
        """运行所有阶段"""
        print("🎯 开始执行完整的Ianvs PIPL隐私保护云边协同提示处理框架")
        print("=" * 80)
        
        self.start_time = datetime.now()
        
        # 执行所有阶段
        success_count = 0
        failed_stages = []
        
        for i, stage in enumerate(self.stages, 1):
            print(f"\n📋 进度: {i}/{len(self.stages)} - {stage}")
            
            if self.run_stage(stage):
                success_count += 1
                print(f"✅ 阶段 {i} 完成")
            else:
                failed_stages.append(stage)
                print(f"❌ 阶段 {i} 失败")
                
                # 询问是否继续
                if i < len(self.stages):
                    print(f"⚠️ 是否继续执行后续阶段？")
                    # 在Colab环境中自动继续
                    print(f"🔄 自动继续执行后续阶段...")
        
        self.end_time = datetime.now()
        total_time = (self.end_time - self.start_time).total_seconds()
        
        # 生成执行报告
        execution_report = {
            "framework_name": "Ianvs PIPL隐私保护云边协同提示处理框架",
            "version": "1.0.0",
            "execution_time": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_seconds": total_time,
                "total_minutes": total_time / 60
            },
            "stages": {
                "total": len(self.stages),
                "successful": success_count,
                "failed": len(failed_stages),
                "success_rate": success_count / len(self.stages)
            },
            "results": self.results,
            "failed_stages": failed_stages,
            "status": "completed" if len(failed_stages) == 0 else "partial"
        }
        
        # 保存执行报告
        report_file = os.path.join(self.base_path, "logs", "complete_execution_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(execution_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 执行完成统计:")
        print(f"  总阶段数: {len(self.stages)}")
        print(f"  成功阶段: {success_count}")
        print(f"  失败阶段: {len(failed_stages)}")
        print(f"  成功率: {success_count/len(self.stages):.1%}")
        print(f"  总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        
        if failed_stages:
            print(f"\n❌ 失败的阶段:")
            for stage in failed_stages:
                print(f"  - {stage}")
        
        print(f"\n📄 执行报告: {report_file}")
        
        return execution_report
    
    def generate_summary_report(self, execution_report: Dict[str, Any]) -> str:
        """生成总结报告"""
        print("\n📋 生成总结报告...")
        
        summary_report = f"""
# Ianvs PIPL隐私保护云边协同提示处理框架执行总结

## 框架信息
- **框架名称**: {execution_report['framework_name']}
- **版本**: {execution_report['version']}
- **执行时间**: {execution_report['execution_time']['start_time']} - {execution_report['execution_time']['end_time']}
- **总耗时**: {execution_report['execution_time']['total_minutes']:.2f} 分钟

## 执行统计
- **总阶段数**: {execution_report['stages']['total']}
- **成功阶段**: {execution_report['stages']['successful']}
- **失败阶段**: {execution_report['stages']['failed']}
- **成功率**: {execution_report['stages']['success_rate']:.1%}

## 阶段详情
"""
        
        for stage, result in self.results.items():
            status_icon = "✅" if result["status"] == "success" else "❌"
            summary_report += f"- {status_icon} **{stage}**: {result['status']}\n"
        
        if execution_report['failed_stages']:
            summary_report += f"\n## 失败阶段\n"
            for stage in execution_report['failed_stages']:
                summary_report += f"- ❌ {stage}\n"
        
        summary_report += f"""
## 生成的文件
- **执行报告**: {os.path.join(self.base_path, 'logs', 'complete_execution_report.json')}
- **结果目录**: {os.path.join(self.base_path, 'results')}
- **日志目录**: {os.path.join(self.base_path, 'logs')}

## 下一步建议
1. 检查失败阶段的错误信息
2. 查看生成的结果文件
3. 根据需要进行调整和优化
4. 使用StoryManager导出功能查看详细结果

---
*报告生成时间: {datetime.now().isoformat()}*
"""
        
        # 保存总结报告
        summary_file = os.path.join(self.base_path, "logs", "execution_summary.md")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"总结报告已保存: {summary_file}")
        return summary_file

def main():
    """主函数"""
    print("🎯 Ianvs PIPL隐私保护云边协同提示处理框架完整执行")
    print("=" * 80)
    
    try:
        # 1. 初始化运行器
        runner = CompletePIPLFrameworkRunner()
        
        # 2. 运行所有阶段
        execution_report = runner.run_all_stages()
        
        # 3. 生成总结报告
        summary_file = runner.generate_summary_report(execution_report)
        
        # 4. 显示最终结果
        print(f"\n🎉 完整框架执行完成！")
        print(f"执行状态: {execution_report['status']}")
        print(f"成功率: {execution_report['stages']['success_rate']:.1%}")
        print(f"总耗时: {execution_report['execution_time']['total_minutes']:.2f} 分钟")
        print(f"执行报告: {os.path.join(runner.base_path, 'logs', 'complete_execution_report.json')}")
        print(f"总结报告: {summary_file}")
        
        if execution_report['status'] == 'completed':
            print(f"\n🎊 恭喜！所有阶段执行成功！")
            print(f"现在可以查看生成的结果文件和使用StoryManager导出功能。")
        else:
            print(f"\n⚠️ 部分阶段执行失败，请检查错误信息并重新运行失败的阶段。")
        
        return execution_report
        
    except Exception as e:
        print(f"❌ 框架执行失败: {e}")
        logger.error(f"框架执行失败: {e}")
        return None

if __name__ == "__main__":
    execution_report = main()
    if execution_report:
        print(f"\n🎯 框架执行完成，状态: {execution_report['status']}")
    else:
        print(f"\n❌ 框架执行失败")
