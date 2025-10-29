"""
PIPL框架自定义异常类

提供完整的异常处理机制，包括：
- 基础异常类
- 隐私相关异常
- 合规性异常
- 配置异常
- 模型加载异常
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PIPLException(Exception):
    """PIPL框架基础异常类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "PIPL_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        
        # 记录异常
        logger.error(f"PIPL Exception [{self.error_code}]: {message}")
        if self.details:
            logger.error(f"Exception details: {self.details}")


class PrivacyBudgetExceededException(PIPLException):
    """隐私预算超限异常"""
    
    def __init__(self, current_budget: float, requested_budget: float, session_id: Optional[str] = None):
        message = f"Privacy budget exceeded. Current: {current_budget}, Requested: {requested_budget}"
        details = {
            'current_budget': current_budget,
            'requested_budget': requested_budget,
            'session_id': session_id,
            'budget_deficit': requested_budget - current_budget
        }
        super().__init__(message, "PRIVACY_BUDGET_EXCEEDED", details)


class ComplianceViolationException(PIPLException):
    """合规性违规异常"""
    
    def __init__(self, violation_type: str, description: str, severity: str = "high", 
                 regulation: str = "PIPL", details: Optional[Dict[str, Any]] = None):
        message = f"Compliance violation [{violation_type}]: {description}"
        violation_details = {
            'violation_type': violation_type,
            'description': description,
            'severity': severity,
            'regulation': regulation,
            'timestamp': datetime.now().isoformat()
        }
        if details:
            violation_details.update(details)
        
        super().__init__(message, "COMPLIANCE_VIOLATION", violation_details)


class ModelLoadException(PIPLException):
    """模型加载异常"""
    
    def __init__(self, model_name: str, error_details: str, model_type: str = "unknown"):
        message = f"Failed to load model '{model_name}': {error_details}"
        details = {
            'model_name': model_name,
            'model_type': model_type,
            'error_details': error_details
        }
        super().__init__(message, "MODEL_LOAD_ERROR", details)


class ConfigurationException(PIPLException):
    """配置异常"""
    
    def __init__(self, config_key: str, expected_type: str, actual_value: Any, 
                 config_section: Optional[str] = None):
        message = f"Configuration error for '{config_key}': expected {expected_type}, got {type(actual_value).__name__}"
        details = {
            'config_key': config_key,
            'config_section': config_section,
            'expected_type': expected_type,
            'actual_type': type(actual_value).__name__,
            'actual_value': str(actual_value)
        }
        super().__init__(message, "CONFIGURATION_ERROR", details)


class DataProcessingException(PIPLException):
    """数据处理异常"""
    
    def __init__(self, operation: str, data_type: str, error_details: str, 
                 data_size: Optional[int] = None):
        message = f"Data processing error in '{operation}' for {data_type}: {error_details}"
        details = {
            'operation': operation,
            'data_type': data_type,
            'data_size': data_size,
            'error_details': error_details
        }
        super().__init__(message, "DATA_PROCESSING_ERROR", details)


class EncryptionException(PIPLException):
    """加密异常"""
    
    def __init__(self, encryption_type: str, operation: str, error_details: str):
        message = f"Encryption error in {encryption_type} {operation}: {error_details}"
        details = {
            'encryption_type': encryption_type,
            'operation': operation,
            'error_details': error_details
        }
        super().__init__(message, "ENCRYPTION_ERROR", details)


class AuditLogException(PIPLException):
    """审计日志异常"""
    
    def __init__(self, operation: str, log_file: str, error_details: str):
        message = f"Audit log error in '{operation}' for file '{log_file}': {error_details}"
        details = {
            'operation': operation,
            'log_file': log_file,
            'error_details': error_details
        }
        super().__init__(message, "AUDIT_LOG_ERROR", details)


class ValidationException(PIPLException):
    """验证异常"""
    
    def __init__(self, validation_type: str, field_name: str, value: Any, 
                 validation_rule: str, error_details: str):
        message = f"Validation failed for {validation_type} '{field_name}': {error_details}"
        details = {
            'validation_type': validation_type,
            'field_name': field_name,
            'value': str(value),
            'validation_rule': validation_rule,
            'error_details': error_details
        }
        super().__init__(message, "VALIDATION_ERROR", details)


class ResourceException(PIPLException):
    """资源异常"""
    
    def __init__(self, resource_type: str, resource_name: str, operation: str, 
                 error_details: str):
        message = f"Resource error for {resource_type} '{resource_name}' in {operation}: {error_details}"
        details = {
            'resource_type': resource_type,
            'resource_name': resource_name,
            'operation': operation,
            'error_details': error_details
        }
        super().__init__(message, "RESOURCE_ERROR", details)


class NetworkException(PIPLException):
    """网络异常"""
    
    def __init__(self, endpoint: str, operation: str, error_details: str, 
                 status_code: Optional[int] = None):
        message = f"Network error for endpoint '{endpoint}' in {operation}: {error_details}"
        details = {
            'endpoint': endpoint,
            'operation': operation,
            'status_code': status_code,
            'error_details': error_details
        }
        super().__init__(message, "NETWORK_ERROR", details)


class TimeoutException(PIPLException):
    """超时异常"""
    
    def __init__(self, operation: str, timeout_seconds: float, error_details: str):
        message = f"Timeout error in '{operation}' after {timeout_seconds}s: {error_details}"
        details = {
            'operation': operation,
            'timeout_seconds': timeout_seconds,
            'error_details': error_details
        }
        super().__init__(message, "TIMEOUT_ERROR", details)


class RateLimitException(PIPLException):
    """速率限制异常"""
    
    def __init__(self, operation: str, rate_limit: int, current_rate: float, 
                 retry_after: Optional[int] = None):
        message = f"Rate limit exceeded for '{operation}': {current_rate}/{rate_limit}"
        details = {
            'operation': operation,
            'rate_limit': rate_limit,
            'current_rate': current_rate,
            'retry_after': retry_after
        }
        super().__init__(message, "RATE_LIMIT_EXCEEDED", details)


# 异常处理工具函数
def handle_exception(func):
    """异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PIPLException as e:
            # PIPL框架异常，直接重新抛出
            raise e
        except Exception as e:
            # 其他异常，包装为PIPL异常
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise PIPLException(
                f"Unexpected error in {func.__name__}: {str(e)}",
                "UNEXPECTED_ERROR",
                {'function': func.__name__, 'original_error': str(e)}
            )
    return wrapper


def safe_execute(func, *args, **kwargs):
    """安全执行函数，捕获并处理异常"""
    try:
        return func(*args, **kwargs)
    except PIPLException as e:
        logger.error(f"PIPL Exception in {func.__name__}: {e.message}")
        return {
            'success': False,
            'error': e.error_code,
            'message': e.message,
            'details': e.details
        }
    except Exception as e:
        logger.error(f"Unexpected error in {func.__name__}: {e}")
        return {
            'success': False,
            'error': 'UNEXPECTED_ERROR',
            'message': f"Unexpected error: {str(e)}",
            'details': {'function': func.__name__, 'original_error': str(e)}
        }


def validate_required_fields(data: Dict[str, Any], required_fields: list) -> None:
    """验证必需字段"""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValidationException(
            "required_fields",
            "data",
            data,
            "all_required_fields_present",
            f"Missing required fields: {missing_fields}"
        )


def validate_field_type(value: Any, expected_type: type, field_name: str) -> None:
    """验证字段类型"""
    if not isinstance(value, expected_type):
        raise ValidationException(
            "field_type",
            field_name,
            value,
            f"type_{expected_type.__name__}",
            f"Expected {expected_type.__name__}, got {type(value).__name__}"
        )


def validate_range(value: float, min_val: float, max_val: float, field_name: str) -> None:
    """验证数值范围"""
    if not (min_val <= value <= max_val):
        raise ValidationException(
            "range",
            field_name,
            value,
            f"range_{min_val}_{max_val}",
            f"Value {value} is not in range [{min_val}, {max_val}]"
        )


# 异常恢复策略
class ExceptionRecoveryStrategy:
    """异常恢复策略"""
    
    @staticmethod
    def retry_on_network_error(max_retries: int = 3, delay: float = 1.0):
        """网络错误重试策略"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                import time
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except NetworkException as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Network error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            delay *= 2  # 指数退避
                        else:
                            raise e
                return None
            return wrapper
        return decorator
    
    @staticmethod
    def fallback_on_model_error(fallback_func):
        """模型错误回退策略"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except ModelLoadException as e:
                    logger.warning(f"Model load failed, using fallback: {e.message}")
                    return fallback_func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def skip_on_validation_error():
        """验证错误跳过策略"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except ValidationException as e:
                    logger.warning(f"Validation failed, skipping operation: {e.message}")
                    return {'success': False, 'skipped': True, 'reason': e.message}
            return wrapper
        return decorator
