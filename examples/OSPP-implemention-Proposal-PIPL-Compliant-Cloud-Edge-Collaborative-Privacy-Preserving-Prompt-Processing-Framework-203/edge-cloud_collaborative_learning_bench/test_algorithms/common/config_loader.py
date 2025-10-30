"""
配置管理和环境变量支持

提供完整的配置管理功能，包括：
- 环境变量加载
- 配置文件解析
- 配置验证
- 配置合并
- 配置热更新
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .exceptions import ConfigurationException, ValidationException
from .error_handling import validate_config, safe_file_operation

logger = logging.getLogger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """配置文件监控处理器"""
    
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.last_modified = {}
    
    def on_modified(self, event):
        """文件修改事件处理"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if file_path in self.config_loader.watched_files:
            # 检查文件是否真的被修改
            current_mtime = os.path.getmtime(file_path)
            if file_path not in self.last_modified or current_mtime > self.last_modified[file_path]:
                self.last_modified[file_path] = current_mtime
                logger.info(f"Config file modified: {file_path}")
                self.config_loader.reload_config(file_path)


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: Optional[str] = None, 
                 auto_reload: bool = False,
                 required_sections: Optional[List[str]] = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
            auto_reload: 是否启用自动重载
            required_sections: 必需的配置节
        """
        self.config_path = config_path
        self.auto_reload = auto_reload
        self.required_sections = required_sections or []
        self.config = {}
        self.watched_files = set()
        self.observer = None
        self.file_handler = None
        self._lock = threading.RLock()
        
        # 加载环境变量
        self._load_environment()
        
        # 加载配置文件
        if config_path:
            self._load_config_file()
        
        # 启动文件监控
        if auto_reload and config_path:
            self._start_file_watcher()
    
    def _load_environment(self):
        """加载环境变量"""
        logger.info("Loading environment variables...")
        
        # 从.env文件加载（如果存在）
        env_file = Path('.env')
        if env_file.exists():
            self._load_env_file(env_file)
        
        # 加载系统环境变量
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
            },
            'error_handling': {
                'max_errors_per_hour': int(os.getenv('MAX_ERRORS_PER_HOUR', '100')),
                'circuit_breaker_threshold': int(os.getenv('CIRCUIT_BREAKER_THRESHOLD', '10')),
                'circuit_breaker_timeout': int(os.getenv('CIRCUIT_BREAKER_TIMEOUT', '300'))
            }
        })
        
        logger.info("Environment variables loaded successfully")
    
    def _load_env_file(self, env_file: Path):
        """加载.env文件"""
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # 移除引号
                        value = value.strip('"\'')
                        os.environ[key.strip()] = value
            logger.info(f"Loaded .env file: {env_file}")
        except Exception as e:
            logger.warning(f"Failed to load .env file: {e}")
    
    def _load_config_file(self):
        """加载配置文件"""
        if not self.config_path or not os.path.exists(self.config_path):
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
                
                # 添加到监控列表
                if self.auto_reload:
                    self.watched_files.add(self.config_path)
                
                logger.info(f"Config file loaded: {self.config_path}")
                
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            raise ConfigurationException(
                "config_file",
                "valid_yaml_or_json",
                self.config_path,
                "config_loader"
            )
    
    def _merge_config(self, base_config: Dict[str, Any], new_config: Dict[str, Any]):
        """合并配置"""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _start_file_watcher(self):
        """启动文件监控"""
        try:
            self.observer = Observer()
            self.file_handler = ConfigFileHandler(self)
            
            # 监控配置文件目录
            config_dir = os.path.dirname(self.config_path) if self.config_path else '.'
            self.observer.schedule(self.file_handler, config_dir, recursive=False)
            self.observer.start()
            
            logger.info(f"File watcher started for: {config_dir}")
        except Exception as e:
            logger.warning(f"Failed to start file watcher: {e}")
    
    def reload_config(self, file_path: str):
        """重新加载配置文件"""
        with self._lock:
            try:
                logger.info(f"Reloading config file: {file_path}")
                
                # 重新加载文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                        new_config = yaml.safe_load(f)
                    elif file_path.endswith('.json'):
                        new_config = json.load(f)
                    else:
                        return
                
                # 验证新配置
                self._validate_config(new_config)
                
                # 更新配置
                self.config = {}
                self._load_environment()
                self._merge_config(self.config, new_config)
                
                logger.info("Config reloaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to reload config: {e}")
    
    def _validate_config(self, config: Dict[str, Any]):
        """验证配置"""
        try:
            # 验证必需节
            if self.required_sections:
                validate_config(config, self.required_sections)
            
            # 验证隐私参数
            if 'privacy' in config:
                privacy_config = config['privacy']
                if 'default_epsilon' in privacy_config:
                    epsilon = privacy_config['default_epsilon']
                    if epsilon <= 0:
                        raise ValidationException(
                            "range",
                            "default_epsilon",
                            epsilon,
                            "positive_value",
                            "Default epsilon must be positive"
                        )
                
                if 'default_delta' in privacy_config:
                    delta = privacy_config['default_delta']
                    if delta <= 0 or delta >= 1:
                        raise ValidationException(
                            "range",
                            "default_delta",
                            delta,
                            "range_0_1",
                            "Default delta must be between 0 and 1"
                        )
            
            # 验证性能参数
            if 'performance' in config:
                perf_config = config['performance']
                if 'max_batch_size' in perf_config:
                    batch_size = perf_config['max_batch_size']
                    if batch_size <= 0:
                        raise ValidationException(
                            "range",
                            "max_batch_size",
                            batch_size,
                            "positive_value",
                            "Max batch size must be positive"
                        )
                
                if 'max_sequence_length' in perf_config:
                    seq_length = perf_config['max_sequence_length']
                    if seq_length <= 0:
                        raise ValidationException(
                            "range",
                            "max_sequence_length",
                            seq_length,
                            "positive_value",
                            "Max sequence length must be positive"
                        )
            
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        with self._lock:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        with self._lock:
            keys = key.split('.')
            config = self.config
            
            # 导航到目标位置
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # 设置值
            config[keys[-1]] = value
            logger.info(f"Config updated: {key} = {value}")
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self._lock:
            return self.config.copy()
    
    def save_config(self, file_path: Optional[str] = None):
        """保存配置到文件"""
        save_path = file_path or self.config_path
        if not save_path:
            raise ConfigurationException(
                "save_path",
                "non_empty_string",
                save_path,
                "config_loader"
            )
        
        def _save_operation():
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                elif save_path.endswith('.json'):
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                else:
                    raise ConfigurationException(
                        "file_format",
                        "yaml_or_json",
                        save_path,
                        "config_loader"
                    )
        
        safe_file_operation("save_config", save_path, _save_operation)
        logger.info(f"Config saved to: {save_path}")
    
    def export_env_file(self, file_path: str = '.env'):
        """导出环境变量文件"""
        def _export_operation():
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# PIPL Framework Environment Variables\n")
                f.write(f"# Generated at: {datetime.now().isoformat()}\n\n")
                
                # 导出API密钥
                api_keys = self.config.get('api_keys', {})
                if api_keys.get('edge_api_key'):
                    f.write(f"EDGE_API_KEY={api_keys['edge_api_key']}\n")
                if api_keys.get('cloud_api_key'):
                    f.write(f"CLOUD_API_KEY={api_keys['cloud_api_key']}\n")
                f.write(f"EDGE_API_BASE={api_keys.get('edge_api_base', 'https://api.openai.com/v1')}\n")
                f.write(f"CLOUD_API_BASE={api_keys.get('cloud_api_base', 'https://api.openai.com/v1')}\n\n")
                
                # 导出隐私配置
                privacy = self.config.get('privacy', {})
                f.write("# Privacy Configuration\n")
                f.write(f"PRIVACY_BUDGET_LIMIT={privacy.get('budget_limit', 10.0)}\n")
                f.write(f"DEFAULT_EPSILON={privacy.get('default_epsilon', 1.2)}\n")
                f.write(f"DEFAULT_DELTA={privacy.get('default_delta', 0.00001)}\n")
                f.write(f"CLIPPING_NORM={privacy.get('clipping_norm', 1.0)}\n\n")
                
                # 导出合规配置
                compliance = self.config.get('compliance', {})
                f.write("# Compliance Configuration\n")
                f.write(f"PIPL_VERSION={compliance.get('pipl_version', '2021')}\n")
                f.write(f"AUDIT_LOG_LEVEL={compliance.get('audit_log_level', 'INFO')}\n")
                f.write(f"COMPLIANCE_MODE={compliance.get('compliance_mode', 'strict')}\n")
                f.write(f"CROSS_BORDER_POLICY={compliance.get('cross_border_policy', 'strict')}\n\n")
                
                # 导出模型配置
                models = self.config.get('models', {})
                f.write("# Model Configuration\n")
                f.write(f"EDGE_MODEL_NAME={models.get('edge_model_name', 'meta-llama/Llama-3-8B-Instruct')}\n")
                f.write(f"CLOUD_MODEL_NAME={models.get('cloud_model_name', 'gpt-4o-mini')}\n")
                f.write(f"NER_MODEL_NAME={models.get('ner_model_name', 'hfl/chinese-bert-wwm-ext')}\n\n")
                
                # 导出性能配置
                performance = self.config.get('performance', {})
                f.write("# Performance Configuration\n")
                f.write(f"MAX_BATCH_SIZE={performance.get('max_batch_size', 32)}\n")
                f.write(f"MAX_SEQUENCE_LENGTH={performance.get('max_sequence_length', 512)}\n")
                f.write(f"ENABLE_QUANTIZATION={str(performance.get('enable_quantization', True)).lower()}\n")
                f.write(f"DEVICE={performance.get('device', 'cpu')}\n\n")
                
                # 导出日志配置
                logging_config = self.config.get('logging', {})
                f.write("# Logging Configuration\n")
                f.write(f"LOG_LEVEL={logging_config.get('log_level', 'INFO')}\n")
                f.write(f"LOG_FILE={logging_config.get('log_file', 'logs/pipl_framework.log')}\n")
                f.write(f"AUDIT_LOG_DIR={logging_config.get('audit_log_dir', 'audit_logs')}\n\n")
                
                # 导出错误处理配置
                error_handling = self.config.get('error_handling', {})
                f.write("# Error Handling Configuration\n")
                f.write(f"MAX_ERRORS_PER_HOUR={error_handling.get('max_errors_per_hour', 100)}\n")
                f.write(f"CIRCUIT_BREAKER_THRESHOLD={error_handling.get('circuit_breaker_threshold', 10)}\n")
                f.write(f"CIRCUIT_BREAKER_TIMEOUT={error_handling.get('circuit_breaker_timeout', 300)}\n")
        
        safe_file_operation("export_env", file_path, _export_operation)
        logger.info(f"Environment variables exported to: {file_path}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        with self._lock:
            return {
                'config_path': self.config_path,
                'auto_reload': self.auto_reload,
                'required_sections': self.required_sections,
                'watched_files': list(self.watched_files),
                'sections': list(self.config.keys()),
                'last_updated': datetime.now().isoformat()
            }
    
    def close(self):
        """关闭配置加载器"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("File watcher stopped")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 全局配置加载器实例
_global_config_loader = None

def get_global_config_loader() -> ConfigLoader:
    """获取全局配置加载器"""
    global _global_config_loader
    if _global_config_loader is None:
        _global_config_loader = ConfigLoader()
    return _global_config_loader

def set_global_config_loader(config_loader: ConfigLoader):
    """设置全局配置加载器"""
    global _global_config_loader
    _global_config_loader = config_loader

def get_config(key: str, default: Any = None) -> Any:
    """获取全局配置值"""
    return get_global_config_loader().get(key, default)

def set_config(key: str, value: Any):
    """设置全局配置值"""
    get_global_config_loader().set(key, value)
