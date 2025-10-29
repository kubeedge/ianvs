#!/usr/bin/env python3
"""
测试文件清理脚本

自动清理项目中的测试文件，合并相关文件，删除重复和临时文件
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

class TestFileCleaner:
    """测试文件清理器"""
    
    def __init__(self, base_path):
        """初始化清理器"""
        self.base_path = Path(base_path)
        self.backup_dir = self.base_path / "test_backup"
        self.cleanup_log = []
        
        # 创建备份目录
        self.backup_dir.mkdir(exist_ok=True)
        
        print(f"测试文件清理器初始化完成: {self.base_path}")
    
    def backup_files(self):
        """备份所有测试文件"""
        print("\n备份测试文件...")
        
        test_files = [
            "test_*.py",
            "test_*.yaml", 
            "test_*.md",
            "*test*.ipynb",
            "*test*.json"
        ]
        
        backed_up = 0
        for pattern in test_files:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file():
                    backup_path = self.backup_dir / file_path.name
                    shutil.copy2(file_path, backup_path)
                    backed_up += 1
                    self.cleanup_log.append(f"备份: {file_path.name}")
        
        print(f"已备份 {backed_up} 个测试文件到 {self.backup_dir}")
    
    def merge_privacy_tests(self):
        """合并隐私模块测试文件"""
        print("\n合并隐私模块测试文件...")
        
        privacy_tests = [
            "test_pii_detector.py",
            "test_differential_privacy.py", 
            "test_pipl_compliance.py"
        ]
        
        merged_content = []
        merged_content.append('#!/usr/bin/env python3\n"""隐私模块综合测试"""\n')
        merged_content.append('import os\nimport sys\nimport json\nimport time\nimport logging\n')
        merged_content.append('from datetime import datetime\n\n')
        
        for test_file in privacy_tests:
            test_path = self.base_path / test_file
            if test_path.exists():
                with open(test_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 提取测试函数
                    lines = content.split('\n')
                    in_test_function = False
                    test_functions = []
                    
                    for line in lines:
                        if line.strip().startswith('def test_'):
                            in_test_function = True
                            test_functions.append(line)
                        elif in_test_function and line.strip() and not line.startswith(' '):
                            in_test_function = False
                        elif in_test_function:
                            test_functions.append(line)
                    
                    if test_functions:
                        merged_content.append(f'\n# === {test_file} ===\n')
                        merged_content.extend(test_functions)
                        merged_content.append('\n')
        
        # 保存合并后的文件
        merged_file = self.base_path / "test_privacy_modules.py"
        with open(merged_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(merged_content))
        
        # 删除原始文件
        for test_file in privacy_tests:
            test_path = self.base_path / test_file
            if test_path.exists():
                test_path.unlink()
                self.cleanup_log.append(f"删除: {test_file}")
        
        print(f"隐私模块测试已合并为: {merged_file.name}")
    
    def merge_system_tests(self):
        """合并系统模块测试文件"""
        print("\n合并系统模块测试文件...")
        
        system_tests = [
            "test_config_management.py",
            "test_error_handling.py"
        ]
        
        merged_content = []
        merged_content.append('#!/usr/bin/env python3\n"""系统模块综合测试"""\n')
        merged_content.append('import os\nimport sys\nimport json\nimport time\nimport logging\n')
        merged_content.append('from datetime import datetime\n\n')
        
        for test_file in system_tests:
            test_path = self.base_path / test_file
            if test_path.exists():
                with open(test_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 提取测试函数
                    lines = content.split('\n')
                    in_test_function = False
                    test_functions = []
                    
                    for line in lines:
                        if line.strip().startswith('def test_'):
                            in_test_function = True
                            test_functions.append(line)
                        elif in_test_function and line.strip() and not line.startswith(' '):
                            in_test_function = False
                        elif in_test_function:
                            test_functions.append(line)
                    
                    if test_functions:
                        merged_content.append(f'\n# === {test_file} ===\n')
                        merged_content.extend(test_functions)
                        merged_content.append('\n')
        
        # 保存合并后的文件
        merged_file = self.base_path / "test_system_modules.py"
        with open(merged_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(merged_content))
        
        # 删除原始文件
        for test_file in system_tests:
            test_path = self.base_path / test_file
            if test_path.exists():
                test_path.unlink()
                self.cleanup_log.append(f"删除: {test_file}")
        
        print(f"系统模块测试已合并为: {merged_file.name}")
    
    def merge_colab_tests(self):
        """合并Colab测试文件"""
        print("\n合并Colab测试文件...")
        
        colab_tests = [
            "colab_dataset_test.py",
            "colab_ianvs_test.py"
        ]
        
        merged_content = []
        merged_content.append('#!/usr/bin/env python3\n"""Colab集成综合测试"""\n')
        merged_content.append('import os\nimport sys\nimport json\nimport time\nimport logging\n')
        merged_content.append('from datetime import datetime\n\n')
        
        for test_file in colab_tests:
            test_path = self.base_path / test_file
            if test_path.exists():
                with open(test_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 提取测试函数
                    lines = content.split('\n')
                    in_test_function = False
                    test_functions = []
                    
                    for line in lines:
                        if line.strip().startswith('def test_'):
                            in_test_function = True
                            test_functions.append(line)
                        elif in_test_function and line.strip() and not line.startswith(' '):
                            in_test_function = False
                        elif in_test_function:
                            test_functions.append(line)
                    
                    if test_functions:
                        merged_content.append(f'\n# === {test_file} ===\n')
                        merged_content.extend(test_functions)
                        merged_content.append('\n')
        
        # 保存合并后的文件
        merged_file = self.base_path / "test_colab_integration.py"
        with open(merged_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(merged_content))
        
        # 删除原始文件
        for test_file in colab_tests:
            test_path = self.base_path / test_file
            if test_path.exists():
                test_path.unlink()
                self.cleanup_log.append(f"删除: {test_file}")
        
        print(f"Colab测试已合并为: {merged_file.name}")
    
    def merge_unsloth_tests(self):
        """合并Unsloth测试文件"""
        print("\n合并Unsloth测试文件...")
        
        unsloth_tests = [
            "test_colab_unsloth_integration.py",
            "test_colab_unsloth_simple.py"
        ]
        
        merged_content = []
        merged_content.append('#!/usr/bin/env python3\n"""Unsloth集成综合测试"""\n')
        merged_content.append('import os\nimport sys\nimport json\nimport time\nimport logging\n')
        merged_content.append('from datetime import datetime\n\n')
        
        for test_file in unsloth_tests:
            test_path = self.base_path / test_file
            if test_path.exists():
                with open(test_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 提取测试函数
                    lines = content.split('\n')
                    in_test_function = False
                    test_functions = []
                    
                    for line in lines:
                        if line.strip().startswith('def test_'):
                            in_test_function = True
                            test_functions.append(line)
                        elif in_test_function and line.strip() and not line.startswith(' '):
                            in_test_function = False
                        elif in_test_function:
                            test_functions.append(line)
                    
                    if test_functions:
                        merged_content.append(f'\n# === {test_file} ===\n')
                        merged_content.extend(test_functions)
                        merged_content.append('\n')
        
        # 保存合并后的文件
        merged_file = self.base_path / "test_colab_unsloth.py"
        with open(merged_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(merged_content))
        
        # 删除原始文件
        for test_file in unsloth_tests:
            test_path = self.base_path / test_file
            if test_path.exists():
                test_path.unlink()
                self.cleanup_log.append(f"删除: {test_file}")
        
        print(f"Unsloth测试已合并为: {merged_file.name}")
    
    def remove_temporary_files(self):
        """删除临时文件"""
        print("\n删除临时文件...")
        
        temp_files = [
            "simple_comprehensive_test_report.json",
            "test_pipl_modules.py",
            "quick_functional_test.py",
            "simple_comprehensive_test.py"
        ]
        
        removed = 0
        for temp_file in temp_files:
            temp_path = self.base_path / temp_file
            if temp_path.exists():
                temp_path.unlink()
                removed += 1
                self.cleanup_log.append(f"删除临时文件: {temp_file}")
        
        print(f"已删除 {removed} 个临时文件")
    
    def merge_documentation(self):
        """合并文档文件"""
        print("\n合并测试文档...")
        
        doc_files = [
            "TESTING_GUIDE.md",
            "FUNCTIONAL_TESTING_GUIDE.md"
        ]
        
        merged_content = []
        merged_content.append("# 🧪 综合测试指南\n")
        merged_content.append("## 📋 概述\n")
        merged_content.append("本指南包含所有测试相关的文档和说明。\n\n")
        
        for doc_file in doc_files:
            doc_path = self.base_path / doc_file
            if doc_path.exists():
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    merged_content.append(f"## {doc_file}\n")
                    merged_content.append(content)
                    merged_content.append("\n---\n")
        
        # 保存合并后的文档
        merged_doc = self.base_path / "COMPREHENSIVE_TESTING_GUIDE.md"
        with open(merged_doc, 'w', encoding='utf-8') as f:
            f.write('\n'.join(merged_content))
        
        # 删除原始文档
        for doc_file in doc_files:
            doc_path = self.base_path / doc_file
            if doc_path.exists():
                doc_path.unlink()
                self.cleanup_log.append(f"删除文档: {doc_file}")
        
        print(f"测试文档已合并为: {merged_doc.name}")
    
    def generate_cleanup_report(self):
        """生成清理报告"""
        print("\n生成清理报告...")
        
        report = {
            "cleanup_time": datetime.now().isoformat(),
            "backup_directory": str(self.backup_dir),
            "cleanup_log": self.cleanup_log,
            "summary": {
                "files_removed": len([log for log in self.cleanup_log if "删除" in log]),
                "files_merged": len([log for log in self.cleanup_log if "合并" in log]),
                "files_backed_up": len([log for log in self.cleanup_log if "备份" in log])
            }
        }
        
        report_path = self.base_path / "test_cleanup_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"清理报告已保存: {report_path}")
        return report
    
    def run_cleanup(self):
        """运行完整清理流程"""
        print("开始测试文件清理...")
        
        try:
            # 1. 备份文件
            self.backup_files()
            
            # 2. 合并测试文件
            self.merge_privacy_tests()
            self.merge_system_tests()
            self.merge_colab_tests()
            self.merge_unsloth_tests()
            
            # 3. 删除临时文件
            self.remove_temporary_files()
            
            # 4. 合并文档
            self.merge_documentation()
            
            # 5. 生成报告
            report = self.generate_cleanup_report()
            
            print(f"\n测试文件清理完成！")
            print(f"   删除文件: {report['summary']['files_removed']} 个")
            print(f"   合并文件: {report['summary']['files_merged']} 个")
            print(f"   备份文件: {report['summary']['files_backed_up']} 个")
            print(f"   备份目录: {self.backup_dir}")
            
            return report
            
        except Exception as e:
            print(f"清理过程中出现错误: {e}")
            return None

def main():
    """主函数"""
    # 设置基础路径
    base_path = "D:/ianvs/examples/OSPP-implemention-Proposal-PIPL-Compliant-Cloud-Edge-Collaborative-Privacy-Preserving-Prompt-Processing-Framework-203/edge-cloud_collaborative_learning_bench"
    
    # 创建清理器
    cleaner = TestFileCleaner(base_path)
    
    # 运行清理
    report = cleaner.run_cleanup()
    
    if report:
        print("\n测试文件清理成功完成！")
    else:
        print("\n测试文件清理失败！")

if __name__ == "__main__":
    main()
