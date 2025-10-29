# 错误处理和异常管理指南

## 1. 自定义异常类

```python
# exceptions.py
class PIPLException(Exception):
    """PIPL框架基础异常"""
    pass

class PrivacyBudgetExceededException(PIPLException):
    """隐私预算超限异常"""
    pass

class ComplianceViolationException(PIPLException):
    """合规性违规异常"""
    pass

class ModelLoadException(PIPLException):
    """模型加载异常"""
    pass

class ConfigurationException(PIPLException):
    """配置异常"""
    pass
```

## 2. 错误处理装饰器

```python
def handle_errors(func):
    """错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PrivacyBudgetExceededException as e:
            logger.error(f"Privacy budget exceeded: {e}")
            return {'error': 'privacy_budget_exceeded', 'message': str(e)}
        except ComplianceViolationException as e:
            logger.error(f"Compliance violation: {e}")
            return {'error': 'compliance_violation', 'message': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return {'error': 'unexpected_error', 'message': str(e)}
    return wrapper
```

## 3. 配置验证

```python
def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置参数"""
    required_keys = ['privacy_detection', 'privacy_encryption', 'compliance']
    
    for key in required_keys:
        if key not in config:
            raise ConfigurationException(f"Missing required config key: {key}")
    
    # 验证隐私参数
    dp_config = config.get('privacy_encryption', {}).get('differential_privacy', {})
    if 'general' not in dp_config:
        raise ConfigurationException("Missing differential privacy general config")
    
    epsilon = dp_config['general'].get('epsilon', 0)
    if epsilon <= 0:
        raise ConfigurationException("Epsilon must be positive")
    
    return True
```

## 4. 资源管理

```python
class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        self.resources = {}
    
    def register_resource(self, name: str, resource):
        """注册资源"""
        self.resources[name] = resource
    
    def cleanup(self):
        """清理资源"""
        for name, resource in self.resources.items():
            try:
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'close'):
                    resource.close()
            except Exception as e:
                logger.warning(f"Failed to cleanup resource {name}: {e}")
```

## 5. 重试机制

```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1.0, backoff=2.0):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise e
                    
                    logger.warning(f"Attempt {attempts} failed: {e}, retrying in {current_delay}s")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator
```
