# 配置管理指南

## 1. 环境变量配置

```bash
# .env 文件示例
# API配置
EDGE_API_KEY=your_edge_api_key_here
CLOUD_API_KEY=your_cloud_api_key_here
EDGE_API_BASE=https://api.edge.example.com/v1
CLOUD_API_BASE=https://api.openai.com/v1

# 隐私保护配置
PRIVACY_BUDGET_LIMIT=10.0
DEFAULT_EPSILON=1.2
DEFAULT_DELTA=0.00001
CLIPPING_NORM=1.0

# 合规性配置
PIPL_VERSION=2021
AUDIT_LOG_LEVEL=INFO
COMPLIANCE_MODE=strict
CROSS_BORDER_POLICY=strict

# 模型配置
EDGE_MODEL_NAME=meta-llama/Llama-3-8B-Instruct
CLOUD_MODEL_NAME=gpt-4o-mini
NER_MODEL_NAME=hfl/chinese-bert-wwm-ext

# 性能配置
MAX_BATCH_SIZE=32
MAX_SEQUENCE_LENGTH=512
ENABLE_QUANTIZATION=true
DEVICE=cpu

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/pipl_framework.log
AUDIT_LOG_DIR=audit_logs
```

## 2. 配置加载器

```python
# config_loader.py
import os
import yaml
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        self._load_environment()
        self._load_config_file()
    
    def _load_environment(self):
        """加载环境变量"""
        load_dotenv()
        
        self.config.update({
            'api_keys': {
                'edge_api_key': os.getenv('EDGE_API_KEY'),
                'cloud_api_key': os.getenv('CLOUD_API_KEY'),
                'edge_api_base': os.getenv('EDGE_API_BASE', 'https://api.openai.com/v1'),
                'cloud_api_base': os.getenv('CLOUD_API_BASE', 'https://api.openai.com/v1')
            },
            'privacy': {
                'budget_limit': float(os.getenv('PRIVACY_BUDGET_LIMIT', '10.0')),
                'default_epsilon': float(os.getenv('DEFAULT_EPSILON', '1.2')),
                'default_delta': float(os.getenv('DEFAULT_DELTA', '0.00001')),
                'clipping_norm': float(os.getenv('CLIPPING_NORM', '1.0'))
            },
            'compliance': {
                'pipl_version': os.getenv('PIPL_VERSION', '2021'),
                'audit_log_level': os.getenv('AUDIT_LOG_LEVEL', 'INFO'),
                'compliance_mode': os.getenv('COMPLIANCE_MODE', 'strict'),
                'cross_border_policy': os.getenv('CROSS_BORDER_POLICY', 'strict')
            },
            'models': {
                'edge_model_name': os.getenv('EDGE_MODEL_NAME', 'meta-llama/Llama-3-8B-Instruct'),
                'cloud_model_name': os.getenv('CLOUD_MODEL_NAME', 'gpt-4o-mini'),
                'ner_model_name': os.getenv('NER_MODEL_NAME', 'hfl/chinese-bert-wwm-ext')
            },
            'performance': {
                'max_batch_size': int(os.getenv('MAX_BATCH_SIZE', '32')),
                'max_sequence_length': int(os.getenv('MAX_SEQUENCE_LENGTH', '512')),
                'enable_quantization': os.getenv('ENABLE_QUANTIZATION', 'true').lower() == 'true',
                'device': os.getenv('DEVICE', 'cpu')
            },
            'logging': {
                'log_level': os.getenv('LOG_LEVEL', 'INFO'),
                'log_file': os.getenv('LOG_FILE', 'logs/pipl_framework.log'),
                'audit_log_dir': os.getenv('AUDIT_LOG_DIR', 'audit_logs')
            }
        })
    
    def _load_config_file(self):
        """加载配置文件"""
        if not self.config_path:
            return
        
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}")
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    file_config = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {self.config_path}")
                    return
                
                # 合并配置
                self._merge_config(self.config, file_config)
                
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
    
    def _merge_config(self, base_config: Dict[str, Any], new_config: Dict[str, Any]):
        """合并配置"""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.config.copy()
```

## 3. 配置验证

```python
def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置完整性"""
    required_sections = ['api_keys', 'privacy', 'compliance', 'models', 'performance', 'logging']
    
    for section in required_sections:
        if section not in config:
            raise ConfigurationException(f"Missing required config section: {section}")
    
    # 验证API密钥
    if not config['api_keys'].get('edge_api_key'):
        logger.warning("Edge API key not configured")
    
    if not config['api_keys'].get('cloud_api_key'):
        logger.warning("Cloud API key not configured")
    
    # 验证隐私参数
    privacy_config = config['privacy']
    if privacy_config['default_epsilon'] <= 0:
        raise ConfigurationException("Default epsilon must be positive")
    
    if privacy_config['default_delta'] <= 0 or privacy_config['default_delta'] >= 1:
        raise ConfigurationException("Default delta must be between 0 and 1")
    
    return True
```

## 4. 配置使用示例

```python
# 使用配置加载器
config_loader = ConfigLoader('config.yaml')
config = config_loader.get_all()

# 验证配置
validate_config(config)

# 使用配置初始化模块
from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy

dp_config = {
    'differential_privacy': {
        'general': {
            'epsilon': config['privacy']['default_epsilon'],
            'delta': config['privacy']['default_delta'],
            'clipping_norm': config['privacy']['clipping_norm']
        }
    },
    'budget_management': {
        'session_limit': config['privacy']['budget_limit'],
        'rate_limit': 5
    }
}

dp = DifferentialPrivacy(dp_config)
```
