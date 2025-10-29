# 部署和运维指南

## 1. 容器化部署

### 1.1 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p logs audit_logs

# 设置环境变量
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "app.py"]
```

### 1.2 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  pipl-framework:
    build: .
    ports:
      - "8000:8000"
    environment:
      - EDGE_API_KEY=${EDGE_API_KEY}
      - CLOUD_API_KEY=${CLOUD_API_KEY}
      - PRIVACY_BUDGET_LIMIT=10.0
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./audit_logs:/app/audit_logs
      - ./config:/app/config
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
```

## 2. Kubernetes部署

### 2.1 部署配置

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pipl-framework
  labels:
    app: pipl-framework
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pipl-framework
  template:
    metadata:
      labels:
        app: pipl-framework
    spec:
      containers:
      - name: pipl-framework
        image: pipl-framework:latest
        ports:
        - containerPort: 8000
        env:
        - name: EDGE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: edge-api-key
        - name: CLOUD_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: cloud-api-key
        - name: PRIVACY_BUDGET_LIMIT
          value: "10.0"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: audit-logs
          mountPath: /app/audit_logs
      volumes:
      - name: logs
        emptyDir: {}
      - name: audit-logs
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: pipl-framework-service
spec:
  selector:
    app: pipl-framework
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 2.2 配置管理

```yaml
# k8s-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pipl-config
data:
  config.yaml: |
    privacy_detection:
      detection_methods:
        regex_patterns: ["phone", "email", "id_card"]
        entity_types: ["PERSON", "PHONE", "EMAIL", "ID_CARD"]
    
    privacy_encryption:
      differential_privacy:
        general:
          epsilon: 1.2
          delta: 0.00001
          clipping_norm: 1.0
    
    compliance:
      pipl_version: "2021"
      audit_level: "detailed"
      cross_border_policy: "strict"
---
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
data:
  edge-api-key: <base64-encoded-key>
  cloud-api-key: <base64-encoded-key>
```

## 3. 监控和日志

### 3.1 Prometheus监控配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'pipl-framework'
    static_configs:
      - targets: ['pipl-framework:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 3.2 告警规则

```yaml
# alert_rules.yml
groups:
- name: pipl-framework
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: PrivacyBudgetExceeded
    expr: privacy_budget_remaining < 1.0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Privacy budget exceeded"
      description: "Privacy budget remaining is {{ $value }}"

  - alert: HighMemoryUsage
    expr: memory_usage_percent > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}%"
```

### 3.3 日志配置

```python
# logging_config.py
import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging():
    """设置日志配置"""
    
    # 创建日志目录
    log_dir = os.getenv('LOG_DIR', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置根日志器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                os.path.join(log_dir, 'pipl_framework.log'),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # 配置审计日志
    audit_logger = logging.getLogger('audit')
    audit_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'audit.log'),
        maxBytes=10*1024*1024,
        backupCount=10
    )
    audit_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(message)s')
    )
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    
    # 配置错误日志
    error_logger = logging.getLogger('error')
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_logger.addHandler(error_handler)
```

## 4. 性能优化

### 4.1 缓存配置

```python
# cache_config.py
import redis
from functools import wraps
import json
import hashlib

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, redis_url='redis://localhost:6379'):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1小时
    
    def cache_result(self, ttl=None):
        """缓存结果装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # 尝试从缓存获取
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
                
                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                self.redis_client.setex(
                    cache_key, 
                    ttl or self.default_ttl, 
                    json.dumps(result, default=str)
                )
                
                return result
            return wrapper
        return decorator
    
    def _generate_cache_key(self, func_name, args, kwargs):
        """生成缓存键"""
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
```

### 4.2 异步处理

```python
# async_processor.py
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import queue

class AsyncProcessor:
    """异步处理器"""
    
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = asyncio.Queue()
    
    async def process_batch(self, tasks):
        """批量异步处理"""
        results = []
        
        # 创建任务
        futures = []
        for task in tasks:
            future = asyncio.create_task(self._process_single(task))
            futures.append(future)
        
        # 等待所有任务完成
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        return results
    
    async def _process_single(self, task):
        """处理单个任务"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行CPU密集型任务
        result = await loop.run_in_executor(
            self.executor, 
            self._cpu_intensive_task, 
            task
        )
        
        return result
    
    def _cpu_intensive_task(self, task):
        """CPU密集型任务"""
        # 实际的PII检测或差分隐私处理
        pass
```

## 5. 安全配置

### 5.1 安全头配置

```python
# security_config.py
from flask import Flask
from flask_talisman import Talisman

def setup_security(app: Flask):
    """设置安全配置"""
    
    # 配置安全头
    Talisman(app, {
        'force_https': True,
        'strict_transport_security': True,
        'strict_transport_security_max_age': 31536000,
        'content_security_policy': {
            'default-src': "'self'",
            'script-src': "'self' 'unsafe-inline'",
            'style-src': "'self' 'unsafe-inline'",
            'img-src': "'self' data: https:",
        },
        'referrer_policy': 'strict-origin-when-cross-origin'
    })
    
    # 配置CORS
    from flask_cors import CORS
    CORS(app, origins=['https://yourdomain.com'])
    
    # 配置速率限制
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=["1000 per hour"]
    )
```

### 5.2 数据加密

```python
# encryption_config.py
from cryptography.fernet import Fernet
import base64
import os

class DataEncryption:
    """数据加密类"""
    
    def __init__(self, key=None):
        if key is None:
            key = os.getenv('ENCRYPTION_KEY')
            if key is None:
                key = Fernet.generate_key()
        
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """加密数据"""
        encrypted_data = self.cipher.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密数据"""
        decoded_data = base64.b64decode(encrypted_data.encode())
        decrypted_data = self.cipher.decrypt(decoded_data)
        return decrypted_data.decode()
```

## 6. 部署脚本

### 6.1 自动化部署脚本

```bash
#!/bin/bash
# deploy.sh

set -e

# 配置变量
APP_NAME="pipl-framework"
VERSION=${1:-latest}
REGISTRY="your-registry.com"
NAMESPACE="pipl"

echo "开始部署 $APP_NAME:$VERSION"

# 构建镜像
echo "构建Docker镜像..."
docker build -t $REGISTRY/$APP_NAME:$VERSION .

# 推送镜像
echo "推送镜像到仓库..."
docker push $REGISTRY/$APP_NAME:$VERSION

# 更新Kubernetes部署
echo "更新Kubernetes部署..."
kubectl set image deployment/$APP_NAME $APP_NAME=$REGISTRY/$APP_NAME:$VERSION -n $NAMESPACE

# 等待部署完成
echo "等待部署完成..."
kubectl rollout status deployment/$APP_NAME -n $NAMESPACE

# 验证部署
echo "验证部署..."
kubectl get pods -n $NAMESPACE -l app=$APP_NAME

echo "部署完成！"
```

### 6.2 健康检查脚本

```bash
#!/bin/bash
# health_check.sh

APP_URL="http://localhost:8000"
MAX_RETRIES=5
RETRY_INTERVAL=10

check_health() {
    local url=$1
    local retries=$2
    
    if [ $retries -eq 0 ]; then
        echo "健康检查失败"
        exit 1
    fi
    
    echo "检查健康状态: $url (剩余重试次数: $retries)"
    
    if curl -f -s $url/health > /dev/null; then
        echo "健康检查通过"
        return 0
    else
        echo "健康检查失败，等待 $RETRY_INTERVAL 秒后重试..."
        sleep $RETRY_INTERVAL
        check_health $url $((retries-1))
    fi
}

check_health $APP_URL $MAX_RETRIES
```

## 7. 运维监控

### 7.1 系统监控

```python
# system_monitor.py
import psutil
import time
import logging

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_system_metrics(self):
        """获取系统指标"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict(),
            'timestamp': time.time()
        }
    
    def check_system_health(self):
        """检查系统健康状态"""
        metrics = self.get_system_metrics()
        
        alerts = []
        
        if metrics['cpu_percent'] > 80:
            alerts.append('High CPU usage')
        
        if metrics['memory_percent'] > 80:
            alerts.append('High memory usage')
        
        if metrics['disk_percent'] > 90:
            alerts.append('High disk usage')
        
        if alerts:
            self.logger.warning(f"System alerts: {alerts}")
        
        return {
            'status': 'healthy' if not alerts else 'warning',
            'alerts': alerts,
            'metrics': metrics
        }
```

这个部署指南提供了完整的部署和运维支持，包括容器化、Kubernetes部署、监控、日志、性能优化、安全配置等各个方面。
