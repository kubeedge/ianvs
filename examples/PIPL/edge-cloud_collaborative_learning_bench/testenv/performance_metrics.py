"""
Performance Metrics for Privacy-Preserving LLM Evaluation

This module implements comprehensive performance evaluation metrics including:
- Latency and throughput measurements
- Resource utilization tracking
- End-to-end performance analysis
- Edge-cloud collaboration efficiency
"""

import logging
import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
import queue

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Comprehensive performance evaluation for privacy-preserving LLM systems.
    
    Tracks and measures various performance aspects including latency,
    throughput, resource utilization, and system efficiency.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize performance metrics collector.
        
        Args:
            config: Configuration dictionary for performance tracking
        """
        self.config = config or {}
        
        # Performance tracking
        self.metrics_history = []
        self.current_session = None
        self.resource_monitor = None
        self.monitoring_active = False
        
        # Performance thresholds
        self.latency_thresholds = {
            'excellent': 1.0,  # seconds
            'good': 2.0,
            'acceptable': 5.0,
            'poor': 10.0
        }
        
        self.throughput_thresholds = {
            'excellent': 20,  # requests/second
            'good': 10,
            'acceptable': 5,
            'poor': 1
        }
        
        logger.info("Performance Metrics collector initialized")
    
    def start_session(self, session_id: str) -> bool:
        """
        Start a new performance monitoring session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            bool: True if session started successfully
        """
        try:
            self.current_session = {
                'session_id': session_id,
                'start_time': datetime.now(),
                'metrics': [],
                'resource_usage': [],
                'requests_processed': 0,
                'errors': 0
            }
            
            # Start resource monitoring
            self._start_resource_monitoring()
            
            logger.info(f"Performance monitoring session started: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start performance session: {e}")
            return False
    
    def end_session(self) -> Dict[str, Any]:
        """
        End current performance monitoring session.
        
        Returns:
            dict: Session performance summary
        """
        if not self.current_session:
            return {'error': 'No active session'}
        
        try:
            # Stop resource monitoring
            self._stop_resource_monitoring()
            
            # Calculate session summary
            session_summary = self._calculate_session_summary()
            
            # Store session results
            self.metrics_history.append(session_summary)
            
            # Clear current session
            self.current_session = None
            
            logger.info(f"Performance monitoring session ended. Summary: {session_summary}")
            return session_summary
            
        except Exception as e:
            logger.error(f"Failed to end performance session: {e}")
            return {'error': str(e)}
    
    def record_request(self, request_data: Dict[str, Any]) -> bool:
        """
        Record performance metrics for a single request.
        
        Args:
            request_data: Dictionary containing request performance data
            
        Returns:
            bool: True if successfully recorded
        """
        if not self.current_session:
            logger.warning("No active session for recording request")
            return False
        
        try:
            # Extract timing information
            timing_data = self._extract_timing_data(request_data)
            
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(timing_data)
            
            # Record request metrics
            request_metrics = {
                'timestamp': datetime.now(),
                'request_id': request_data.get('request_id', f"req_{int(time.time())}"),
                'timing': timing_data,
                'derived_metrics': derived_metrics,
                'success': request_data.get('success', True),
                'error_type': request_data.get('error_type', None)
            }
            
            # Add to session
            self.current_session['metrics'].append(request_metrics)
            self.current_session['requests_processed'] += 1
            
            if not request_metrics['success']:
                self.current_session['errors'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record request metrics: {e}")
            return False
    
    def _extract_timing_data(self, request_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract timing information from request data."""
        timing = {}
        
        # Extract various timing measurements
        timing['total_latency'] = request_data.get('total_latency', 0.0)
        timing['edge_processing_time'] = request_data.get('edge_processing_time', 0.0)
        timing['cloud_processing_time'] = request_data.get('cloud_processing_time', 0.0)
        timing['network_latency'] = request_data.get('network_latency', 0.0)
        timing['privacy_processing_time'] = request_data.get('privacy_processing_time', 0.0)
        timing['inference_time'] = request_data.get('inference_time', 0.0)
        
        # Calculate missing timing if possible
        if timing['total_latency'] == 0.0:
            timing['total_latency'] = sum([
                timing['edge_processing_time'],
                timing['cloud_processing_time'],
                timing['network_latency']
            ])
        
        return timing
    
    def _calculate_derived_metrics(self, timing_data: Dict[str, float]) -> Dict[str, Any]:
        """Calculate derived performance metrics."""
        derived = {}
        
        # Calculate efficiency metrics
        total_time = timing_data['total_latency']
        if total_time > 0:
            derived['edge_efficiency'] = timing_data['edge_processing_time'] / total_time
            derived['cloud_efficiency'] = timing_data['cloud_processing_time'] / total_time
            derived['network_efficiency'] = timing_data['network_latency'] / total_time
            derived['privacy_overhead'] = timing_data['privacy_processing_time'] / total_time
        else:
            derived['edge_efficiency'] = 0.0
            derived['cloud_efficiency'] = 0.0
            derived['network_efficiency'] = 0.0
            derived['privacy_overhead'] = 0.0
        
        # Calculate performance grades
        derived['latency_grade'] = self._grade_latency(total_time)
        derived['efficiency_score'] = self._calculate_efficiency_score(timing_data)
        
        return derived
    
    def _grade_latency(self, latency: float) -> str:
        """Grade latency performance."""
        if latency <= self.latency_thresholds['excellent']:
            return 'excellent'
        elif latency <= self.latency_thresholds['good']:
            return 'good'
        elif latency <= self.latency_thresholds['acceptable']:
            return 'acceptable'
        else:
            return 'poor'
    
    def _calculate_efficiency_score(self, timing_data: Dict[str, float]) -> float:
        """Calculate overall efficiency score."""
        # Weight different components
        weights = {
            'edge_processing_time': 0.3,
            'cloud_processing_time': 0.3,
            'network_latency': 0.2,
            'privacy_processing_time': 0.2
        }
        
        total_time = timing_data['total_latency']
        if total_time == 0:
            return 0.0
        
        # Calculate weighted efficiency (lower processing time = higher efficiency)
        efficiency = 0.0
        for component, weight in weights.items():
            if component in timing_data:
                component_efficiency = 1.0 - (timing_data[component] / total_time)
                efficiency += weight * max(0.0, component_efficiency)
        
        return min(1.0, max(0.0, efficiency))
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.resource_monitor = threading.Thread(target=self._monitor_resources)
        self.resource_monitor.daemon = True
        self.resource_monitor.start()
    
    def _stop_resource_monitoring(self):
        """Stop background resource monitoring."""
        self.monitoring_active = False
        if self.resource_monitor:
            self.resource_monitor.join(timeout=1.0)
    
    def _monitor_resources(self):
        """Background resource monitoring thread."""
        while self.monitoring_active:
            try:
                if self.current_session:
                    resource_data = self._collect_resource_usage()
                    self.current_session['resource_usage'].append(resource_data)
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(1.0)
    
    def _collect_resource_usage(self) -> Dict[str, Any]:
        """Collect current resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network I/O (if available)
            try:
                net_io = psutil.net_io_counters()
                network_data = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            except:
                network_data = {}
            
            return {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_mb': memory_used_mb,
                'disk_percent': disk_percent,
                'network': network_data
            }
            
        except Exception as e:
            logger.error(f"Failed to collect resource usage: {e}")
            return {'timestamp': datetime.now(), 'error': str(e)}
    
    def _calculate_session_summary(self) -> Dict[str, Any]:
        """Calculate comprehensive session performance summary."""
        if not self.current_session:
            return {'error': 'No session data'}
        
        session = self.current_session
        metrics = session['metrics']
        
        if not metrics:
            return {
                'session_id': session['session_id'],
                'duration': 0,
                'requests_processed': 0,
                'error': 'No metrics recorded'
            }
        
        # Calculate timing statistics
        latencies = [m['timing']['total_latency'] for m in metrics if m['success']]
        edge_times = [m['timing']['edge_processing_time'] for m in metrics if m['success']]
        cloud_times = [m['timing']['cloud_processing_time'] for m in metrics if m['success']]
        network_times = [m['timing']['network_latency'] for m in metrics if m['success']]
        privacy_times = [m['timing']['privacy_processing_time'] for m in metrics if m['success']]
        
        # Calculate throughput
        duration = (datetime.now() - session['start_time']).total_seconds()
        throughput = len(metrics) / duration if duration > 0 else 0
        
        # Calculate resource usage statistics
        resource_stats = self._calculate_resource_statistics(session['resource_usage'])
        
        # Calculate performance grades
        avg_latency = np.mean(latencies) if latencies else 0
        latency_grade = self._grade_latency(avg_latency)
        throughput_grade = self._grade_throughput(throughput)
        
        # Calculate efficiency metrics
        efficiency_scores = [m['derived_metrics']['efficiency_score'] for m in metrics if m['success']]
        avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0
        
        # Calculate error rate
        error_rate = session['errors'] / session['requests_processed'] if session['requests_processed'] > 0 else 0
        
        summary = {
            'session_id': session['session_id'],
            'start_time': session['start_time'].isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': duration,
            'requests_processed': session['requests_processed'],
            'errors': session['errors'],
            'error_rate': error_rate,
            
            # Timing statistics
            'latency_stats': {
                'mean': np.mean(latencies) if latencies else 0,
                'median': np.median(latencies) if latencies else 0,
                'std': np.std(latencies) if latencies else 0,
                'min': np.min(latencies) if latencies else 0,
                'max': np.max(latencies) if latencies else 0,
                'p95': np.percentile(latencies, 95) if latencies else 0,
                'p99': np.percentile(latencies, 99) if latencies else 0
            },
            
            'component_times': {
                'edge_processing': {
                    'mean': np.mean(edge_times) if edge_times else 0,
                    'std': np.std(edge_times) if edge_times else 0
                },
                'cloud_processing': {
                    'mean': np.mean(cloud_times) if cloud_times else 0,
                    'std': np.std(cloud_times) if cloud_times else 0
                },
                'network_latency': {
                    'mean': np.mean(network_times) if network_times else 0,
                    'std': np.std(network_times) if network_times else 0
                },
                'privacy_processing': {
                    'mean': np.mean(privacy_times) if privacy_times else 0,
                    'std': np.std(privacy_times) if privacy_times else 0
                }
            },
            
            # Performance metrics
            'throughput': {
                'requests_per_second': throughput,
                'grade': throughput_grade
            },
            
            'latency_grade': latency_grade,
            'efficiency_score': avg_efficiency,
            
            # Resource usage
            'resource_usage': resource_stats,
            
            # Performance assessment
            'performance_assessment': {
                'overall_grade': self._calculate_overall_grade(latency_grade, throughput_grade, avg_efficiency),
                'bottlenecks': self._identify_bottlenecks(edge_times, cloud_times, network_times, privacy_times),
                'recommendations': self._generate_performance_recommendations(avg_latency, throughput, error_rate)
            }
        }
        
        return summary
    
    def _calculate_resource_statistics(self, resource_usage: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate resource usage statistics."""
        if not resource_usage:
            return {}
        
        # Extract metrics
        cpu_values = [r['cpu_percent'] for r in resource_usage if 'cpu_percent' in r]
        memory_values = [r['memory_percent'] for r in resource_usage if 'memory_percent' in r]
        memory_mb_values = [r['memory_used_mb'] for r in resource_usage if 'memory_used_mb' in r]
        
        return {
            'cpu_usage': {
                'mean': np.mean(cpu_values) if cpu_values else 0,
                'max': np.max(cpu_values) if cpu_values else 0,
                'std': np.std(cpu_values) if cpu_values else 0
            },
            'memory_usage': {
                'percent_mean': np.mean(memory_values) if memory_values else 0,
                'percent_max': np.max(memory_values) if memory_values else 0,
                'mb_mean': np.mean(memory_mb_values) if memory_mb_values else 0,
                'mb_max': np.max(memory_mb_values) if memory_mb_values else 0
            }
        }
    
    def _grade_throughput(self, throughput: float) -> str:
        """Grade throughput performance."""
        if throughput >= self.throughput_thresholds['excellent']:
            return 'excellent'
        elif throughput >= self.throughput_thresholds['good']:
            return 'good'
        elif throughput >= self.throughput_thresholds['acceptable']:
            return 'acceptable'
        else:
            return 'poor'
    
    def _calculate_overall_grade(self, latency_grade: str, throughput_grade: str, efficiency: float) -> str:
        """Calculate overall performance grade."""
        # Grade mapping to numeric values
        grade_values = {'excellent': 4, 'good': 3, 'acceptable': 2, 'poor': 1}
        
        latency_score = grade_values.get(latency_grade, 1)
        throughput_score = grade_values.get(throughput_grade, 1)
        efficiency_score = int(efficiency * 4) + 1  # Convert 0-1 to 1-4
        
        # Calculate weighted average
        overall_score = (latency_score * 0.4 + throughput_score * 0.4 + efficiency_score * 0.2)
        
        # Convert back to grade
        if overall_score >= 3.5:
            return 'excellent'
        elif overall_score >= 2.5:
            return 'good'
        elif overall_score >= 1.5:
            return 'acceptable'
        else:
            return 'poor'
    
    def _identify_bottlenecks(self, edge_times: List[float], cloud_times: List[float], 
                            network_times: List[float], privacy_times: List[float]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Calculate average times
        avg_edge = np.mean(edge_times) if edge_times else 0
        avg_cloud = np.mean(cloud_times) if cloud_times else 0
        avg_network = np.mean(network_times) if network_times else 0
        avg_privacy = np.mean(privacy_times) if privacy_times else 0
        
        total_time = avg_edge + avg_cloud + avg_network + avg_privacy
        
        if total_time > 0:
            # Check if any component takes more than 40% of total time
            if avg_edge / total_time > 0.4:
                bottlenecks.append('edge_processing')
            if avg_cloud / total_time > 0.4:
                bottlenecks.append('cloud_processing')
            if avg_network / total_time > 0.4:
                bottlenecks.append('network_latency')
            if avg_privacy / total_time > 0.4:
                bottlenecks.append('privacy_processing')
        
        return bottlenecks
    
    def _generate_performance_recommendations(self, avg_latency: float, throughput: float, error_rate: float) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        if avg_latency > self.latency_thresholds['acceptable']:
            recommendations.append("Consider optimizing processing pipeline to reduce latency")
            recommendations.append("Review edge-cloud communication efficiency")
        
        if throughput < self.throughput_thresholds['acceptable']:
            recommendations.append("Implement request batching to improve throughput")
            recommendations.append("Consider horizontal scaling for edge processing")
        
        if error_rate > 0.05:  # 5% error rate threshold
            recommendations.append("Investigate and fix error sources to improve reliability")
            recommendations.append("Implement better error handling and retry mechanisms")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable ranges")
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary across all sessions."""
        if not self.metrics_history:
            return {'error': 'No performance data available'}
        
        # Aggregate statistics across all sessions
        all_latencies = []
        all_throughputs = []
        all_efficiencies = []
        total_requests = 0
        total_errors = 0
        
        for session in self.metrics_history:
            if 'latency_stats' in session:
                all_latencies.append(session['latency_stats']['mean'])
            if 'throughput' in session:
                all_throughputs.append(session['throughput']['requests_per_second'])
            if 'efficiency_score' in session:
                all_efficiencies.append(session['efficiency_score'])
            
            total_requests += session.get('requests_processed', 0)
            total_errors += session.get('errors', 0)
        
        return {
            'total_sessions': len(self.metrics_history),
            'total_requests': total_requests,
            'total_errors': total_errors,
            'overall_error_rate': total_errors / total_requests if total_requests > 0 else 0,
            'average_latency': np.mean(all_latencies) if all_latencies else 0,
            'average_throughput': np.mean(all_throughputs) if all_throughputs else 0,
            'average_efficiency': np.mean(all_efficiencies) if all_efficiencies else 0,
            'performance_trend': self._analyze_performance_trend()
        }
    
    def _analyze_performance_trend(self) -> str:
        """Analyze performance trend across sessions."""
        if len(self.metrics_history) < 2:
            return 'insufficient_data'
        
        # Analyze latency trend
        recent_sessions = self.metrics_history[-5:]  # Last 5 sessions
        latencies = [s['latency_stats']['mean'] for s in recent_sessions if 'latency_stats' in s]
        
        if len(latencies) >= 2:
            if latencies[-1] < latencies[0] * 0.9:  # 10% improvement
                return 'improving'
            elif latencies[-1] > latencies[0] * 1.1:  # 10% degradation
                return 'degrading'
            else:
                return 'stable'
        
        return 'stable'

