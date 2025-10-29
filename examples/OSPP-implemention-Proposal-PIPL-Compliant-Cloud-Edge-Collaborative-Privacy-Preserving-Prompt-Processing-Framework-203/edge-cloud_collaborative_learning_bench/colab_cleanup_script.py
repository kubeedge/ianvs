#!/usr/bin/env python3
"""
Colab部署文件清理脚本

删除所有Colab部署相关的文件
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

class ColabFileCleaner:
    """Colab文件清理器"""
    
    def __init__(self, base_path):
        """初始化清理器"""
        self.base_path = Path(base_path)
        self.backup_dir = self.base_path / "colab_backup"
        self.cleanup_log = []
        
        # 创建备份目录
        self.backup_dir.mkdir(exist_ok=True)
        
        print(f"Colab文件清理器初始化完成: {self.base_path}")
    
    def backup_files(self):
        """备份所有Colab文件"""
        print("\n备份Colab文件...")
        
        colab_files = [
            "colab_*.py",
            "Colab_*.ipynb", 
            "COLAB_*.md",
            "*colab*.yaml",
            "*Colab*.yaml"
        ]
        
        backed_up = 0
        for pattern in colab_files:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file():
                    backup_path = self.backup_dir / file_path.name
                    shutil.copy2(file_path, backup_path)
                    backed_up += 1
                    self.cleanup_log.append(f"备份: {file_path.name}")
        
        print(f"已备份 {backed_up} 个Colab文件到 {self.backup_dir}")
    
    def delete_colab_files(self):
        """删除所有Colab相关文件"""
        print("\n删除Colab文件...")
        
        # 定义要删除的文件模式
        colab_patterns = [
            "colab_*.py",
            "Colab_*.ipynb",
            "COLAB_*.md", 
            "*colab*.yaml",
            "*Colab*.yaml",
            "FINAL_COLAB_*.md",
            "START_COLAB_*.md",
            "UNSLOTH_COLAB_*.md",
            "PIPL_*_Colab.ipynb"
        ]
        
        deleted = 0
        for pattern in colab_patterns:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    deleted += 1
                    self.cleanup_log.append(f"删除: {file_path.name}")
        
        print(f"已删除 {deleted} 个Colab文件")
    
    def delete_colab_directories(self):
        """删除Colab相关目录"""
        print("\n删除Colab相关目录...")
        
        # 检查是否有Colab相关的目录需要删除
        colab_dirs = []
        for item in self.base_path.iterdir():
            if item.is_dir() and "colab" in item.name.lower():
                colab_dirs.append(item)
        
        deleted_dirs = 0
        for colab_dir in colab_dirs:
            shutil.rmtree(colab_dir)
            deleted_dirs += 1
            self.cleanup_log.append(f"删除目录: {colab_dir.name}")
        
        print(f"已删除 {deleted_dirs} 个Colab相关目录")
    
    def generate_cleanup_report(self):
        """生成清理报告"""
        print("\n生成清理报告...")
        
        report = {
            "cleanup_time": datetime.now().isoformat(),
            "backup_directory": str(self.backup_dir),
            "cleanup_log": self.cleanup_log,
            "summary": {
                "files_backed_up": len([log for log in self.cleanup_log if "备份" in log]),
                "files_deleted": len([log for log in self.cleanup_log if "删除:" in log]),
                "directories_deleted": len([log for log in self.cleanup_log if "删除目录:" in log])
            }
        }
        
        report_path = self.base_path / "colab_cleanup_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"清理报告已保存: {report_path}")
        return report
    
    def run_cleanup(self):
        """运行完整清理流程"""
        print("开始Colab文件清理...")
        
        try:
            # 1. 备份文件
            self.backup_files()
            
            # 2. 删除Colab文件
            self.delete_colab_files()
            
            # 3. 删除Colab目录
            self.delete_colab_directories()
            
            # 4. 生成报告
            report = self.generate_cleanup_report()
            
            print(f"\nColab文件清理完成！")
            print(f"   备份文件: {report['summary']['files_backed_up']} 个")
            print(f"   删除文件: {report['summary']['files_deleted']} 个")
            print(f"   删除目录: {report['summary']['directories_deleted']} 个")
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
    cleaner = ColabFileCleaner(base_path)
    
    # 运行清理
    report = cleaner.run_cleanup()
    
    if report:
        print("\nColab文件清理成功完成！")
    else:
        print("\nColab文件清理失败！")

if __name__ == "__main__":
    main()
