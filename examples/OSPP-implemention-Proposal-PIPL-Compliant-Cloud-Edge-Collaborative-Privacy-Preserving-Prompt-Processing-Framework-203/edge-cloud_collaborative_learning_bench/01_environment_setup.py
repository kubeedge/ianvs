#!/usr/bin/env python3
"""
阶段1: 环境准备和检查

在Colab环境中准备运行环境，检查系统要求，设置基础配置
"""

import os
import sys
import platform
import psutil
import subprocess
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python版本过低，需要3.8+")
        return False
    
    # 检查内存
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    print(f"系统内存: {memory_gb:.1f} GB")
    
    if memory_gb < 8:
        print("⚠️ 内存不足，建议8GB+")
    
    # 检查磁盘空间
    disk = psutil.disk_usage('/')
    disk_gb = disk.free / (1024**3)
    print(f"可用磁盘空间: {disk_gb:.1f} GB")
    
    if disk_gb < 10:
        print("⚠️ 磁盘空间不足，建议10GB+")
    
    # 检查CUDA（如果可用）
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
        else:
            print("CUDA不可用，将使用CPU")
    except ImportError:
        print("PyTorch未安装")
    
    print("✅ 系统要求检查完成")
    return True

def setup_working_directory():
    """设置工作目录"""
    print("\n📁 设置工作目录...")
    
    # 创建主工作目录
    base_path = "/content/ianvs_pipl_framework"
    os.makedirs(base_path, exist_ok=True)
    os.chdir(base_path)
    
    # 创建子目录
    directories = [
        "data",
        "models", 
        "results",
        "logs",
        "configs",
        "scripts"
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)
        print(f"创建目录: {directory}")
    
    print(f"工作目录: {os.getcwd()}")
    return base_path

def setup_environment_variables():
    """设置环境变量"""
    print("\n🔧 设置环境变量...")
    
    env_vars = {
        "IANVS_HOME": "/content/ianvs_pipl_framework",
        "PYTHONPATH": "/content/ianvs_pipl_framework",
        "CUDA_VISIBLE_DEVICES": "0",
        "TOKENIZERS_PARALLELISM": "false"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置 {key} = {value}")
    
    print("✅ 环境变量设置完成")

def check_network_connectivity():
    """检查网络连接"""
    print("\n🌐 检查网络连接...")
    
    try:
        import requests
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            print("✅ 网络连接正常")
            return True
        else:
            print("❌ 网络连接异常")
            return False
    except Exception as e:
        print(f"❌ 网络连接失败: {e}")
        return False

def initialize_logging():
    """初始化日志系统"""
    print("\n📝 初始化日志系统...")
    
    log_dir = "/content/ianvs_pipl_framework/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件
    log_file = os.path.join(log_dir, f"ianvs_pipl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Ianvs PIPL框架日志系统初始化完成")
    print(f"日志文件: {log_file}")
    
    return log_file

def create_config_file():
    """创建配置文件"""
    print("\n⚙️ 创建配置文件...")
    
    config_content = """
# Ianvs PIPL框架配置文件
framework:
  name: "Ianvs PIPL Privacy Protection Framework"
  version: "1.0.0"
  environment: "colab"

# 模型配置
models:
  edge_model:
    name: "Qwen2.5-7B-Edge"
    type: "edge"
    quantization: "4bit"
    optimization: "unsloth"
  
  cloud_model:
    name: "Qwen2.5-7B-Cloud"
    type: "cloud"
    quantization: "8bit"
    optimization: "unsloth"

# 隐私保护配置
privacy:
  pipl_compliance: true
  privacy_budget: 1.2
  epsilon: 1.2
  delta: 0.00001
  clipping_norm: 1.0

# 数据集配置
dataset:
  name: "ChnSentiCorp-Lite"
  format: "jsonl"
  train_samples: 1000
  val_samples: 200
  test_samples: 200

# 性能配置
performance:
  batch_size: 8
  max_length: 512
  num_workers: 2
  device: "cuda"
"""
    
    config_file = "/content/ianvs_pipl_framework/configs/framework_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"配置文件: {config_file}")
    return config_file

def main():
    """主函数"""
    print("🚀 阶段1: 环境准备和检查")
    print("=" * 50)
    
    try:
        # 1. 检查系统要求
        if not check_system_requirements():
            print("❌ 系统要求检查失败")
            return False
        
        # 2. 设置工作目录
        base_path = setup_working_directory()
        
        # 3. 设置环境变量
        setup_environment_variables()
        
        # 4. 检查网络连接
        if not check_network_connectivity():
            print("⚠️ 网络连接异常，可能影响模型下载")
        
        # 5. 初始化日志系统
        log_file = initialize_logging()
        
        # 6. 创建配置文件
        config_file = create_config_file()
        
        # 7. 保存环境信息
        env_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "disk_gb": psutil.disk_usage('/').free / (1024**3),
            "working_directory": base_path,
            "log_file": log_file,
            "config_file": config_file
        }
        
        import json
        env_info_file = os.path.join(base_path, "logs", "environment_info.json")
        with open(env_info_file, 'w', encoding='utf-8') as f:
            json.dump(env_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 环境准备完成！")
        print(f"工作目录: {base_path}")
        print(f"日志文件: {log_file}")
        print(f"配置文件: {config_file}")
        print(f"环境信息: {env_info_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 环境准备失败: {e}")
        logger.error(f"环境准备失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 阶段1完成，可以继续执行阶段2")
    else:
        print("\n❌ 阶段1失败，请检查错误信息")
