#!/usr/bin/env python3
"""
阶段2: 依赖安装

安装所有必需的依赖包，包括Ianvs框架、隐私保护模块、可视化工具等
"""

import os
import sys
import subprocess
import importlib
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_package(package_name, import_name=None, upgrade=False):
    """安装单个包"""
    if import_name is None:
        import_name = package_name
    
    try:
        # 检查是否已安装
        importlib.import_module(import_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"📦 安装 {package_name}...")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.append(package_name)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {package_name} 安装成功")
                return True
            else:
                print(f"❌ {package_name} 安装失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {package_name} 安装超时")
            return False
        except Exception as e:
            print(f"❌ {package_name} 安装异常: {e}")
            return False

def install_core_dependencies():
    """安装核心依赖"""
    print("🔧 安装核心依赖...")
    
    core_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        ("tqdm", "tqdm"),
        ("requests", "requests"),
        ("psutil", "psutil"),
        ("pyyaml", "yaml"),
        ("jsonlines", "jsonlines"),
        ("rich", "rich"),
        ("loguru", "loguru")
    ]
    
    success_count = 0
    for package, import_name in core_packages:
        if install_package(package, import_name):
            success_count += 1
    
    print(f"核心依赖安装完成: {success_count}/{len(core_packages)}")
    return success_count == len(core_packages)

def install_nlp_dependencies():
    """安装NLP依赖"""
    print("\n🔤 安装NLP依赖...")
    
    nlp_packages = [
        ("spacy", "spacy"),
        ("jieba", "jieba"),
        ("nltk", "nltk"),
        ("openai", "openai"),
        ("huggingface-hub", "huggingface_hub")
    ]
    
    success_count = 0
    for package, import_name in nlp_packages:
        if install_package(package, import_name):
            success_count += 1
    
    # 下载spacy模型
    try:
        print("📥 下载spacy中文模型...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "zh_core_web_sm"], 
                      capture_output=True, text=True, timeout=300)
        print("✅ spacy中文模型下载完成")
    except Exception as e:
        print(f"⚠️ spacy模型下载失败: {e}")
    
    print(f"NLP依赖安装完成: {success_count}/{len(nlp_packages)}")
    return success_count == len(nlp_packages)

def install_privacy_dependencies():
    """安装隐私保护依赖"""
    print("\n🔒 安装隐私保护依赖...")
    
    privacy_packages = [
        ("opacus", "opacus"),
        ("cryptography", "cryptography"),
        ("membership-inference-attacks", "mia"),
        ("differential-privacy", "differential_privacy")
    ]
    
    success_count = 0
    for package, import_name in privacy_packages:
        if install_package(package, import_name):
            success_count += 1
    
    print(f"隐私保护依赖安装完成: {success_count}/{len(privacy_packages)}")
    return success_count == len(privacy_packages)

def install_ianvs_dependencies():
    """安装Ianvs框架依赖"""
    print("\n🏗️ 安装Ianvs框架依赖...")
    
    # 首先尝试从源码安装Ianvs
    try:
        print("📦 从源码安装Ianvs...")
        
        # 克隆Ianvs仓库（如果不存在）
        if not os.path.exists("/content/ianvs"):
            subprocess.run([
                "git", "clone", 
                "https://github.com/kubeedge/ianvs.git",
                "/content/ianvs"
            ], check=True, timeout=300)
        
        # 安装Ianvs
        os.chdir("/content/ianvs")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                      check=True, timeout=300)
        
        print("✅ Ianvs框架安装成功")
        return True
        
    except Exception as e:
        print(f"⚠️ Ianvs框架安装失败: {e}")
        print("尝试安装Ianvs相关依赖...")
        
        # 安装Ianvs相关依赖
        ianvs_packages = [
            ("prettytable", "prettytable"),
            ("onnx", "onnx"),
            ("onnxruntime", "onnxruntime"),
            ("pydantic", "pydantic"),
            ("click", "click")
        ]
        
        success_count = 0
        for package, import_name in ianvs_packages:
            if install_package(package, import_name):
                success_count += 1
        
        print(f"Ianvs相关依赖安装完成: {success_count}/{len(ianvs_packages)}")
        return success_count == len(ianvs_packages)

def install_visualization_dependencies():
    """安装可视化依赖"""
    print("\n📊 安装可视化依赖...")
    
    viz_packages = [
        ("plotly", "plotly"),
        ("dash", "dash"),
        ("bokeh", "bokeh"),
        ("altair", "altair"),
        ("wordcloud", "wordcloud")
    ]
    
    success_count = 0
    for package, import_name in viz_packages:
        if install_package(package, import_name):
            success_count += 1
    
    print(f"可视化依赖安装完成: {success_count}/{len(viz_packages)}")
    return success_count == len(viz_packages)

def verify_installation():
    """验证安装结果"""
    print("\n🔍 验证安装结果...")
    
    # 测试关键模块
    test_modules = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("spacy", "spaCy"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("sklearn", "Scikit-learn"),
        ("requests", "Requests"),
        ("yaml", "PyYAML")
    ]
    
    success_count = 0
    for module, name in test_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {name} 可用")
            success_count += 1
        except ImportError:
            print(f"❌ {name} 不可用")
    
    print(f"模块验证完成: {success_count}/{len(test_modules)}")
    return success_count == len(test_modules)

def create_requirements_file():
    """创建requirements.txt文件"""
    print("\n📝 创建requirements.txt文件...")
    
    requirements = """# Ianvs PIPL框架依赖列表
# 核心依赖
torch>=2.0.0
transformers>=4.30.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
tqdm>=4.60.0
requests>=2.25.0
psutil>=5.8.0
pyyaml>=6.0
jsonlines>=3.0.0
rich>=12.0.0
loguru>=0.6.0

# NLP依赖
spacy>=3.4.0
jieba>=0.42.1
nltk>=3.7
openai>=0.27.0
huggingface-hub>=0.15.0

# 隐私保护依赖
opacus>=1.3.0
cryptography>=3.4.8
membership-inference-attacks>=0.1.0

# 可视化依赖
dash>=2.0.0
bokeh>=2.4.0
altair>=4.2.0
wordcloud>=1.8.0

# Ianvs相关依赖
prettytable>=3.0.0
onnx>=1.12.0
onnxruntime>=1.12.0
pydantic>=1.8.0
click>=8.0.0
"""
    
    requirements_file = "/content/ianvs_pipl_framework/requirements.txt"
    with open(requirements_file, 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print(f"requirements.txt文件: {requirements_file}")
    return requirements_file

def main():
    """主函数"""
    print("🚀 阶段2: 依赖安装")
    print("=" * 50)
    
    try:
        # 1. 安装核心依赖
        core_success = install_core_dependencies()
        
        # 2. 安装NLP依赖
        nlp_success = install_nlp_dependencies()
        
        # 3. 安装隐私保护依赖
        privacy_success = install_privacy_dependencies()
        
        # 4. 安装Ianvs框架依赖
        ianvs_success = install_ianvs_dependencies()
        
        # 5. 安装可视化依赖
        viz_success = install_visualization_dependencies()
        
        # 6. 验证安装结果
        verify_success = verify_installation()
        
        # 7. 创建requirements.txt文件
        requirements_file = create_requirements_file()
        
        # 8. 保存安装报告
        installation_report = {
            "timestamp": datetime.now().isoformat(),
            "core_dependencies": core_success,
            "nlp_dependencies": nlp_success,
            "privacy_dependencies": privacy_success,
            "ianvs_dependencies": ianvs_success,
            "visualization_dependencies": viz_success,
            "verification": verify_success,
            "requirements_file": requirements_file
        }
        
        import json
        report_file = "/content/ianvs_pipl_framework/logs/installation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(installation_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 依赖安装完成！")
        print(f"安装报告: {report_file}")
        print(f"requirements.txt: {requirements_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 依赖安装失败: {e}")
        logger.error(f"依赖安装失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 阶段2完成，可以继续执行阶段3")
    else:
        print("\n❌ 阶段2失败，请检查错误信息")
