"""
Error Handling Utilities and Decorators

Provides comprehensive error handling mechanisms including:
- Error handling decorators
- Retry mechanisms
- Resource management
- Configuration validation
"""

import logging
import time
import functools
import psutil
import os
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
from datetime import datetime, timedelta

from .exceptions import (
    PIPLException, PrivacyBudgetExceededException, ComplianceViolationException,
    ModelLoadException, ConfigurationException, DataProcessingException,
    EncryptionException, AuditLogException, ValidationException,
    ResourceException, NetworkException, TimeoutException, RateLimitException
)

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Error Handler for comprehensive error management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.error_counts = {}
        self.last_error_time = {}
        self.max_errors_per_hour = config.get('max_errors_per_hour', 100)
        self.circuit_breaker_threshold = config.get('circuit_breaker_threshold', 10)
        self.circuit_breaker_timeout = config.get('circuit_breaker_timeout', 300)  # 5 minutes
        self.circuit_breaker_state = {}  # 'closed', 'open', 'half_open'
    
    def should_circuit_break(self, operation: str) -> bool:
        """Check if circuit breaker should be triggered"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Clean up expired error records
        if operation in self.last_error_time:
            if self.last_error_time[operation] < hour_ago:
                self.error_counts[operation] = 0
        
        # 检查错误计数
        error_count = self.error_counts.get(operation, 0)
        if error_count >= self.circuit_breaker_threshold:
            # 检查熔断状态
            if operation not in self.circuit_breaker_state:
                self.circuit_breaker_state[operation] = 'open'
                logger.warning(f"Circuit breaker opened for operation: {operation}")
                return True
            
            if self.circuit_breaker_state[operation] == 'open':
                # 检查是否可以尝试半开状态
                if (operation in self.last_error_time and 
                    now - self.last_error_time[operation] > timedelta(seconds=self.circuit_breaker_timeout)):
                    self.circuit_breaker_state[operation] = 'half_open'
                    logger.info(f"Circuit breaker half-open for operation: {operation}")
                    return False
                return True
        
        return False
    
    def record_error(self, operation: str, error: Exception):
        """记录错误"""
        now = datetime.now()
        self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
        self.last_error_time[operation] = now
        
        logger.error(f"Error recorded for operation {operation}: {error}")
    
    def record_success(self, operation: str):
        """记录成功"""
        if operation in self.circuit_breaker_state:
            self.circuit_breaker_state[operation] = 'closed'
            self.error_counts[operation] = 0
            logger.info(f"Circuit breaker closed for operation: {operation}")


def handle_errors(error_handler: Optional[ErrorHandler] = None, 
                  operation_name: Optional[str] = None,
                  max_retries: int = 0,
                  retry_delay: float = 1.0,
                  exponential_backoff: bool = True):
    """错误处理装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            
            # 检查熔断器
            if error_handler and error_handler.should_circuit_break(op_name):
                raise PIPLException(
                    f"Circuit breaker is open for operation: {op_name}",
                    "CIRCUIT_BREAKER_OPEN",
                    {'operation': op_name}
                )
            
            last_exception = None
            delay = retry_delay
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # 记录成功
                    if error_handler:
                        error_handler.record_success(op_name)
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # 记录错误
                    if error_handler:
                        error_handler.record_error(op_name, e)
                    
                    # 如果是最后一次尝试，直接抛出异常
                    if attempt == max_retries:
                        break
                    
                    # 等待重试
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {op_name}: {e}, retrying in {delay}s")
                        time.sleep(delay)
                        
                        if exponential_backoff:
                            delay *= 2
            
            # 抛出最后一个异常
            if isinstance(last_exception, PIPLException):
                raise last_exception
            else:
                raise PIPLException(
                    f"Operation {op_name} failed after {max_retries + 1} attempts: {str(last_exception)}",
                    "OPERATION_FAILED",
                    {'operation': op_name, 'attempts': max_retries + 1, 'last_error': str(last_exception)}
                )
        
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {current_delay}s")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


def timeout(seconds: float):
    """超时装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutException(func.__name__, seconds, "Operation timed out")
            
            # 设置超时信号
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 恢复原来的信号处理器
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


def rate_limit(calls_per_second: float = 1.0):
    """速率限制装饰器"""
    def decorator(func):
        last_called = [0.0]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            time_since_last_call = now - last_called[0]
            min_interval = 1.0 / calls_per_second
            
            if time_since_last_call < min_interval:
                sleep_time = min_interval - time_since_last_call
                time.sleep(sleep_time)
            
            last_called[0] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


@contextmanager
def resource_manager(resource_name: str, cleanup_func: Optional[Callable] = None):
    """资源管理器上下文"""
    resource = None
    try:
        logger.info(f"Acquiring resource: {resource_name}")
        yield resource
    except Exception as e:
        logger.error(f"Error with resource {resource_name}: {e}")
        raise
    finally:
        if cleanup_func:
            try:
                logger.info(f"Cleaning up resource: {resource_name}")
                cleanup_func(resource)
            except Exception as e:
                logger.error(f"Error cleaning up resource {resource_name}: {e}")


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_threshold = config.get('memory_threshold', 80.0)  # 80%
        self.cpu_threshold = config.get('cpu_threshold', 80.0)  # 80%
        self.disk_threshold = config.get('disk_threshold', 90.0)  # 90%
    
    def check_system_resources(self) -> Dict[str, Any]:
        """检查系统资源"""
        try:
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # 检查阈值
            warnings = []
            if memory_percent > self.memory_threshold:
                warnings.append(f"High memory usage: {memory_percent:.1f}%")
            if cpu_percent > self.cpu_threshold:
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
            if disk_percent > self.disk_threshold:
                warnings.append(f"High disk usage: {disk_percent:.1f}%")
            
            return {
                'memory_percent': memory_percent,
                'cpu_percent': cpu_percent,
                'disk_percent': disk_percent,
                'warnings': warnings,
                'status': 'warning' if warnings else 'healthy'
            }
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def should_throttle(self) -> bool:
        """检查是否应该限流"""
        resources = self.check_system_resources()
        return len(resources.get('warnings', [])) > 0


def validate_config(config: Dict[str, Any], required_sections: List[str]) -> None:
    """验证配置完整性"""
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ConfigurationException(
            "config",
            "dict",
            config,
            "all_required_sections_present"
        )
    
    # 验证隐私参数
    if 'privacy_encryption' in config:
        dp_config = config['privacy_encryption'].get('differential_privacy', {})
        if 'general' in dp_config:
            general_config = dp_config['general']
            epsilon = general_config.get('epsilon', 0)
            delta = general_config.get('delta', 0)
            
            if epsilon <= 0:
                raise ConfigurationException(
                    "epsilon",
                    "positive_float",
                    epsilon,
                    "privacy_encryption.differential_privacy.general"
                )
            
            if delta <= 0 or delta >= 1:
                raise ConfigurationException(
                    "delta",
                    "float_between_0_and_1",
                    delta,
                    "privacy_encryption.differential_privacy.general"
                )


def safe_file_operation(operation: str, file_path: str, func: Callable, *args, **kwargs):
    """安全文件操作"""
    try:
        # 检查文件路径
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        return func(*args, **kwargs)
        
    except PermissionError as e:
        raise ResourceException(
            "file",
            file_path,
            operation,
            f"Permission denied: {e}"
        )
    except FileNotFoundError as e:
        raise ResourceException(
            "file",
            file_path,
            operation,
            f"File not found: {e}"
        )
    except OSError as e:
        raise ResourceException(
            "file",
            file_path,
            operation,
            f"OS error: {e}"
        )


def log_operation_result(operation: str, success: bool, duration: float, 
                        details: Optional[Dict[str, Any]] = None):
    """记录操作结果"""
    level = logging.INFO if success else logging.ERROR
    message = f"Operation '{operation}' {'completed' if success else 'failed'} in {duration:.2f}s"
    
    if details:
        message += f" - Details: {details}"
    
    logger.log(level, message)


def monitor_operation(operation_name: str):
    """操作监控装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            details = {}
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                details['error'] = str(e)
                details['error_type'] = type(e).__name__
                raise
            finally:
                duration = time.time() - start_time
                log_operation_result(operation_name, success, duration, details)
        
        return wrapper
    return decorator


# 全局错误处理器实例
_global_error_handler = None

def get_global_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler({
            'max_errors_per_hour': 100,
            'circuit_breaker_threshold': 10,
            'circuit_breaker_timeout': 300
        })
    return _global_error_handler


def set_global_error_handler(error_handler: ErrorHandler):
    """设置全局错误处理器"""
    global _global_error_handler
    _global_error_handler = error_handler
