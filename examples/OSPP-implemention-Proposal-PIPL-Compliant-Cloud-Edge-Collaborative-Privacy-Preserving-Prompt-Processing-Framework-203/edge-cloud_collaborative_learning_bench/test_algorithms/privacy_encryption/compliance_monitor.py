"""
Compliance Monitor for PIPL and Privacy Regulations

This module provides comprehensive compliance monitoring, audit logging,
and real-time verification for privacy-preserving LLM inference systems.
"""

import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path

# Import sedna ClassFactory for Ianvs integration
try:
    from sedna.common.class_factory import ClassFactory, ClassType
    SEDNA_AVAILABLE = True
except ImportError:
    SEDNA_AVAILABLE = False
    # Create dummy decorators if sedna is not available
    class ClassFactory:
        @staticmethod
        def register(class_type, alias=None):
            def decorator(cls):
                return cls
            return decorator
    class ClassType:
        GENERAL = "general"

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.GENERAL, alias="ComplianceMonitor")
class ComplianceMonitor:
    """
    Monitors and enforces compliance with PIPL and other privacy regulations.
    
    Provides:
    - Real-time compliance checking
    - Comprehensive audit logging
    - Privacy budget tracking
    - Cross-border transmission monitoring
    - Compliance reporting and alerts
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize compliance monitor.
        
        Args:
            config: Configuration dictionary containing compliance parameters
        """
        self.config = config
        self.compliance_config = config.get('compliance', {})
        
        # Compliance settings
        self.pipl_version = self.compliance_config.get('pipl_version', '2021')
        self.audit_level = self.compliance_config.get('audit_level', 'detailed')
        self.cross_border_policy = self.compliance_config.get('cross_border_policy', 'strict')
        self.minimal_necessity = self.compliance_config.get('minimal_necessity', True)
        
        # Audit logging
        self.audit_logs = []
        self.max_log_entries = 10000
        self.log_retention_days = 30
        
        # Compliance tracking
        self.compliance_violations = []
        self.privacy_budget_tracking = {}
        self.cross_border_transmissions = []
        
        # Initialize audit log directory
        self._init_audit_logging()
        
        logger.info("Compliance Monitor initialized")
    
    def _init_audit_logging(self):
        """Initialize audit logging infrastructure."""
        try:
            # Create audit log directory
            self.audit_log_dir = Path("audit_logs")
            self.audit_log_dir.mkdir(exist_ok=True)
            
            # Create session-specific log file
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.audit_log_file = self.audit_log_dir / f"compliance_audit_{session_id}.jsonl"
            
            logger.info(f"Audit logging initialized: {self.audit_log_file}")
            
        except Exception as e:
            logger.error(f"Failed to initialize audit logging: {e}")
            self.audit_log_file = None
    
    def log_audit(self, audit_entry: Dict[str, Any]) -> bool:
        """
        Log an audit entry for compliance tracking.
        
        Args:
            audit_entry: Dictionary containing audit information
            
        Returns:
            bool: True if successfully logged
        """
        try:
            # Add metadata to audit entry
            enriched_entry = self._enrich_audit_entry(audit_entry)
            
            # Add to in-memory logs
            self.audit_logs.append(enriched_entry)
            
            # Write to file if available
            if self.audit_log_file:
                self._write_audit_entry(enriched_entry)
            
            # Check for compliance violations
            self._check_compliance_violations(enriched_entry)
            
            # Cleanup old logs if needed
            self._cleanup_old_logs()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}")
            return False
    
    def _enrich_audit_entry(self, audit_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich audit entry with additional metadata."""
        enriched = audit_entry.copy()
        
        # Add standard metadata
        enriched['audit_id'] = self._generate_audit_id()
        enriched['timestamp'] = datetime.now().isoformat()
        enriched['session_id'] = getattr(self, 'session_id', 'unknown')
        enriched['compliance_version'] = self.pipl_version
        
        # Add hash for integrity verification
        enriched['integrity_hash'] = self._calculate_integrity_hash(enriched)
        
        return enriched
    
    def _generate_audit_id(self) -> str:
        """Generate unique audit ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        random_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        return f"audit_{timestamp}_{random_suffix}"
    
    def _calculate_integrity_hash(self, entry: Dict[str, Any]) -> str:
        """Calculate integrity hash for audit entry."""
        # Create a copy without the hash field for calculation
        entry_copy = {k: v for k, v in entry.items() if k != 'integrity_hash'}
        
        # Sort keys for consistent hashing
        sorted_entry = json.dumps(entry_copy, sort_keys=True, default=str)
        
        return hashlib.sha256(sorted_entry.encode()).hexdigest()
    
    def _write_audit_entry(self, entry: Dict[str, Any]):
        """Write audit entry to file."""
        try:
            with open(self.audit_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")
    
    def _check_compliance_violations(self, audit_entry: Dict[str, Any]):
        """Check for compliance violations in audit entry."""
        violations = []
        
        # Check PIPL compliance
        if not self._check_pipl_compliance(audit_entry):
            violations.append({
                'type': 'pipl_violation',
                'description': 'PIPL compliance check failed',
                'timestamp': audit_entry['timestamp']
            })
        
        # Check cross-border policy
        if not self._check_cross_border_policy(audit_entry):
            violations.append({
                'type': 'cross_border_violation',
                'description': 'Cross-border transmission policy violated',
                'timestamp': audit_entry['timestamp']
            })
        
        # Check minimal necessity principle
        if not self._check_minimal_necessity(audit_entry):
            violations.append({
                'type': 'minimal_necessity_violation',
                'description': 'Minimal necessity principle violated',
                'timestamp': audit_entry['timestamp']
            })
        
        # Add violations to tracking
        if violations:
            self.compliance_violations.extend(violations)
            logger.warning(f"Compliance violations detected: {len(violations)}")
    
    def _check_pipl_compliance(self, audit_entry: Dict[str, Any]) -> bool:
        """Check PIPL compliance for audit entry."""
        # Check if high-sensitivity data is being transmitted cross-border
        privacy_level = audit_entry.get('privacy_level', 'general')
        cross_border = audit_entry.get('cross_border_transmitted', False)
        
        if privacy_level == 'high_sensitivity' and cross_border:
            return False
        
        # Check if proper consent mechanisms are in place
        if cross_border and not audit_entry.get('consent_obtained', False):
            return False
        
        return True
    
    def _check_cross_border_policy(self, audit_entry: Dict[str, Any]) -> bool:
        """Check cross-border transmission policy compliance."""
        if self.cross_border_policy == 'strict':
            # Strict policy: no high-sensitivity data cross-border
            privacy_level = audit_entry.get('privacy_level', 'general')
            cross_border = audit_entry.get('cross_border_transmitted', False)
            
            if privacy_level == 'high_sensitivity' and cross_border:
                return False
        
        return True
    
    def _check_minimal_necessity(self, audit_entry: Dict[str, Any]) -> bool:
        """Check minimal necessity principle compliance."""
        if not self.minimal_necessity:
            return True
        
        # Check if only necessary data is being transmitted
        cross_border = audit_entry.get('cross_border_transmitted', False)
        if cross_border:
            # Verify that only anonymized vectors and minimal tags are transmitted
            payload = audit_entry.get('payload_transmitted', {})
            if payload and 'raw_text' in payload:
                return False
        
        return True
    
    def track_privacy_budget(self, budget_consumed: float, session_id: str) -> Dict[str, Any]:
        """
        Track privacy budget consumption.
        
        Args:
            budget_consumed: Amount of privacy budget consumed
            session_id: Session identifier
            
        Returns:
            dict: Budget tracking information
        """
        if session_id not in self.privacy_budget_tracking:
            self.privacy_budget_tracking[session_id] = {
                'total_consumed': 0.0,
                'queries': 0,
                'start_time': datetime.now(),
                'violations': 0
            }
        
        session_tracking = self.privacy_budget_tracking[session_id]
        session_tracking['total_consumed'] += budget_consumed
        session_tracking['queries'] += 1
        
        # Check for budget violations
        budget_limit = 10.0  # Default limit
        if session_tracking['total_consumed'] > budget_limit:
            session_tracking['violations'] += 1
            logger.warning(f"Privacy budget exceeded for session {session_id}")
        
        return {
            'session_id': session_id,
            'total_consumed': session_tracking['total_consumed'],
            'queries': session_tracking['queries'],
            'violations': session_tracking['violations'],
            'budget_limit': budget_limit,
            'remaining_budget': max(0, budget_limit - session_tracking['total_consumed'])
        }
    
    def log_cross_border_transmission(self, transmission_data: Dict[str, Any]) -> bool:
        """
        Log cross-border data transmission for compliance tracking.
        
        Args:
            transmission_data: Information about the transmission
            
        Returns:
            bool: True if successfully logged
        """
        try:
            transmission_entry = {
                'timestamp': datetime.now().isoformat(),
                'transmission_id': self._generate_audit_id(),
                'data_type': transmission_data.get('data_type', 'unknown'),
                'privacy_level': transmission_data.get('privacy_level', 'general'),
                'payload_size': transmission_data.get('payload_size', 0),
                'destination': transmission_data.get('destination', 'unknown'),
                'consent_obtained': transmission_data.get('consent_obtained', False),
                'anonymization_applied': transmission_data.get('anonymization_applied', False),
                'compliance_verified': transmission_data.get('compliance_verified', False)
            }
            
            self.cross_border_transmissions.append(transmission_entry)
            
            # Log as audit entry
            self.log_audit({
                'event_type': 'cross_border_transmission',
                'transmission_data': transmission_entry
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log cross-border transmission: {e}")
            return False
    
    def generate_compliance_report(self, start_time: Optional[datetime] = None, 
                                 end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report.
        
        Args:
            start_time: Report start time (default: 24 hours ago)
            end_time: Report end time (default: now)
            
        Returns:
            dict: Comprehensive compliance report
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now()
        
        # Filter logs by time range
        filtered_logs = [
            log for log in self.audit_logs
            if start_time <= datetime.fromisoformat(log['timestamp']) <= end_time
        ]
        
        # Calculate compliance metrics
        total_events = len(filtered_logs)
        violations = len(self.compliance_violations)
        violation_rate = violations / total_events if total_events > 0 else 0
        
        # Privacy budget analysis
        budget_analysis = self._analyze_privacy_budget_usage()
        
        # Cross-border transmission analysis
        cross_border_analysis = self._analyze_cross_border_transmissions()
        
        # Generate report
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'total_events': total_events
            },
            'compliance_summary': {
                'overall_compliance_score': max(0, 1.0 - violation_rate),
                'total_violations': violations,
                'violation_rate': violation_rate,
                'compliance_status': 'compliant' if violation_rate < 0.05 else 'non_compliant'
            },
            'privacy_budget_analysis': budget_analysis,
            'cross_border_analysis': cross_border_analysis,
            'violation_details': self.compliance_violations[-10:],  # Last 10 violations
            'recommendations': self._generate_compliance_recommendations(violation_rate)
        }
        
        return report
    
    def _analyze_privacy_budget_usage(self) -> Dict[str, Any]:
        """Analyze privacy budget usage across sessions."""
        if not self.privacy_budget_tracking:
            return {'sessions': 0, 'total_consumed': 0.0}
        
        total_consumed = sum(session['total_consumed'] for session in self.privacy_budget_tracking.values())
        total_queries = sum(session['queries'] for session in self.privacy_budget_tracking.values())
        total_violations = sum(session['violations'] for session in self.privacy_budget_tracking.values())
        
        return {
            'sessions': len(self.privacy_budget_tracking),
            'total_consumed': total_consumed,
            'total_queries': total_queries,
            'total_violations': total_violations,
            'average_per_session': total_consumed / len(self.privacy_budget_tracking) if self.privacy_budget_tracking else 0,
            'average_per_query': total_consumed / total_queries if total_queries > 0 else 0
        }
    
    def _analyze_cross_border_transmissions(self) -> Dict[str, Any]:
        """Analyze cross-border transmission patterns."""
        if not self.cross_border_transmissions:
            return {'transmissions': 0, 'compliant_transmissions': 0}
        
        total_transmissions = len(self.cross_border_transmissions)
        compliant_transmissions = sum(1 for t in self.cross_border_transmissions 
                                    if t.get('compliance_verified', False))
        
        # Analyze by privacy level
        privacy_level_counts = {}
        for transmission in self.cross_border_transmissions:
            level = transmission.get('privacy_level', 'unknown')
            privacy_level_counts[level] = privacy_level_counts.get(level, 0) + 1
        
        return {
            'transmissions': total_transmissions,
            'compliant_transmissions': compliant_transmissions,
            'compliance_rate': compliant_transmissions / total_transmissions if total_transmissions > 0 else 0,
            'privacy_level_distribution': privacy_level_counts
        }
    
    def _generate_compliance_recommendations(self, violation_rate: float) -> List[str]:
        """Generate compliance recommendations based on violation rate."""
        recommendations = []
        
        if violation_rate > 0.1:
            recommendations.append("High violation rate detected. Review privacy protection mechanisms.")
            recommendations.append("Consider implementing stricter access controls.")
        
        if violation_rate > 0.05:
            recommendations.append("Moderate violation rate. Enhance audit logging and monitoring.")
            recommendations.append("Review cross-border transmission policies.")
        
        if violation_rate > 0.01:
            recommendations.append("Low violation rate. Continue current practices with minor improvements.")
        
        # Budget-specific recommendations
        budget_analysis = self._analyze_privacy_budget_usage()
        if budget_analysis.get('total_violations', 0) > 0:
            recommendations.append("Privacy budget violations detected. Implement stricter budget controls.")
        
        return recommendations
    
    def _cleanup_old_logs(self):
        """Cleanup old audit logs to prevent memory issues."""
        if len(self.audit_logs) > self.max_log_entries:
            # Keep only the most recent entries
            self.audit_logs = self.audit_logs[-self.max_log_entries:]
        
        # Remove old violation records
        cutoff_time = datetime.now() - timedelta(days=self.log_retention_days)
        self.compliance_violations = [
            v for v in self.compliance_violations
            if datetime.fromisoformat(v['timestamp']) > cutoff_time
        ]
    
    def save_audit_logs(self, filepath: Optional[str] = None) -> bool:
        """
        Save audit logs to file.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            bool: True if successfully saved
        """
        try:
            if filepath is None:
                filepath = f"compliance_audit_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'audit_logs': self.audit_logs,
                    'compliance_violations': self.compliance_violations,
                    'privacy_budget_tracking': self.privacy_budget_tracking,
                    'cross_border_transmissions': self.cross_border_transmissions,
                    'export_timestamp': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Audit logs saved to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audit logs: {e}")
            return False
    
    def get_realtime_compliance_status(self) -> Dict[str, Any]:
        """
        Get real-time compliance status.
        
        Returns:
            dict: Current compliance status
        """
        recent_violations = len([
            v for v in self.compliance_violations
            if datetime.fromisoformat(v['timestamp']) > datetime.now() - timedelta(hours=1)
        ])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'recent_violations_1h': recent_violations,
            'total_violations': len(self.compliance_violations),
            'active_sessions': len(self.privacy_budget_tracking),
            'compliance_status': 'healthy' if recent_violations == 0 else 'warning',
            'last_audit_entry': self.audit_logs[-1]['timestamp'] if self.audit_logs else None
        }
    
    def check_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check data compliance with PIPL and privacy regulations.
        
        Args:
            data: Data dictionary containing type, content, and metadata
            
        Returns:
            Compliance check result with status and recommendations
        """
        compliance_result = {
            'status': 'compliant',
            'violations': [],
            'recommendations': [],
            'risk_level': 'low',
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract data information
        data_type = data.get('type', 'unknown')
        content = data.get('content', '')
        cross_border = data.get('cross_border', False)
        risk_level = data.get('risk_level', 'low')
        
        # Check data type compliance
        if data_type in ['personal_info', 'sensitive_info', 'biometric_data']:
            compliance_result['status'] = 'requires_protection'
            compliance_result['risk_level'] = 'high'
            compliance_result['recommendations'].append('Apply encryption')
            compliance_result['recommendations'].append('Implement access controls')
        
        # Check cross-border transmission
        if cross_border:
            if self.cross_border_policy == 'strict':
                compliance_result['status'] = 'requires_encryption'
                compliance_result['violations'].append('Cross-border transmission detected')
                compliance_result['recommendations'].append('Use cross-border encryption')
                compliance_result['recommendations'].append('Obtain explicit consent')
            else:
                compliance_result['recommendations'].append('Review cross-border policy')
        
        # Check risk level
        if risk_level in ['high', 'critical']:
            compliance_result['risk_level'] = risk_level
            if compliance_result['status'] == 'compliant':
                compliance_result['status'] = 'requires_protection'
            compliance_result['recommendations'].append('Apply enhanced protection measures')
        
        # Check minimal necessity principle
        if self.minimal_necessity and len(content) > 1000:
            compliance_result['recommendations'].append('Review data minimization requirements')
        
        # Log compliance check
        self.log_audit({
            'action': 'compliance_check',
            'data_type': data_type,
            'status': compliance_result['status'],
            'risk_level': compliance_result['risk_level'],
            'cross_border': cross_border
        })
        
        return compliance_result
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Returns:
            List of audit log entries
        """
        try:
            if os.path.exists(self.audit_log_file):
                with open(self.audit_log_file, 'r', encoding='utf-8') as f:
                    return [json.loads(line) for line in f if line.strip()]
            return []
        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")
            return []
    
    def log_operation(self, operation: Dict[str, Any]) -> bool:
        """
        Log an operation to audit trail.
        
        Args:
            operation: Operation details dictionary
            
        Returns:
            True if logging successful, False otherwise
        """
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'operation_id': operation.get('operation_id', 'unknown'),
                'operation_type': operation.get('operation_type', 'unknown'),
                'user_id': operation.get('user_id', 'unknown'),
                'data_type': operation.get('data_type', 'unknown'),
                'status': 'logged',
                'details': operation.get('details', {})
            }
            
            # Add to in-memory logs
            self.audit_logs.append(log_entry)
            
            # Write to file
            with open(self.audit_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            # Maintain log size limit
            if len(self.audit_logs) > self.max_log_entries:
                self.audit_logs = self.audit_logs[-self.max_log_entries:]
            
            return True
        except Exception as e:
            logger.error(f"Failed to log operation: {e}")
            return False
    
    def get_audit_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive audit report.
        
        Returns:
            Audit report with statistics and analysis
        """
        try:
            audit_logs = self.get_audit_log()
            
            if not audit_logs:
                return {
                    'total_entries': 0,
                    'period': 'No data available',
                    'summary': 'No audit entries found'
                }
            
            # Calculate time period
            timestamps = [entry.get('timestamp', '') for entry in audit_logs if entry.get('timestamp')]
            if timestamps:
                start_time = min(timestamps)
                end_time = max(timestamps)
                period = f"{start_time} to {end_time}"
            else:
                period = "Unknown"
            
            # Analyze operation types
            operation_types = {}
            for entry in audit_logs:
                op_type = entry.get('operation_type', 'unknown')
                operation_types[op_type] = operation_types.get(op_type, 0) + 1
            
            # Analyze data types
            data_types = {}
            for entry in audit_logs:
                data_type = entry.get('data_type', 'unknown')
                data_types[data_type] = data_types.get(data_type, 0) + 1
            
            # Analyze compliance status
            compliance_statuses = {}
            for entry in audit_logs:
                status = entry.get('status', 'unknown')
                compliance_statuses[status] = compliance_statuses.get(status, 0) + 1
            
            return {
                'total_entries': len(audit_logs),
                'period': period,
                'operation_types': operation_types,
                'data_types': data_types,
                'compliance_statuses': compliance_statuses,
                'recent_entries': audit_logs[-10:] if len(audit_logs) > 10 else audit_logs,
                'generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to generate audit report: {e}")
            return {
                'error': str(e),
                'total_entries': 0,
                'period': 'Error',
                'summary': 'Failed to generate report'
            }
    
    def get_compliance_statistics(self) -> Dict[str, Any]:
        """
        Get compliance statistics and metrics.
        
        Returns:
            Compliance statistics dictionary
        """
        try:
            audit_logs = self.get_audit_log()
            
            if not audit_logs:
                return {
                    'total_operations': 0,
                    'compliance_rate': 0.0,
                    'violation_count': 0,
                    'risk_distribution': {},
                    'summary': 'No data available'
                }
            
            # Calculate compliance metrics
            total_operations = len(audit_logs)
            compliant_operations = len([entry for entry in audit_logs if entry.get('status') == 'compliant'])
            violation_count = total_operations - compliant_operations
            compliance_rate = compliant_operations / total_operations if total_operations > 0 else 0.0
            
            # Analyze risk distribution
            risk_levels = {}
            for entry in audit_logs:
                risk_level = entry.get('risk_level', 'unknown')
                risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
            
            # Analyze cross-border transmissions
            cross_border_count = len([entry for entry in audit_logs if entry.get('cross_border', False)])
            
            # Recent activity (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_entries = [
                entry for entry in audit_logs 
                if entry.get('timestamp') and 
                datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')) > recent_cutoff
            ]
            
            return {
                'total_operations': total_operations,
                'compliant_operations': compliant_operations,
                'violation_count': violation_count,
                'compliance_rate': compliance_rate,
                'risk_distribution': risk_levels,
                'cross_border_transmissions': cross_border_count,
                'recent_activity_24h': len(recent_entries),
                'active_sessions': len(self.privacy_budget_tracking),
                'privacy_budget_utilization': self._calculate_budget_utilization(),
                'generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to generate compliance statistics: {e}")
            return {
                'error': str(e),
                'total_operations': 0,
                'compliance_rate': 0.0,
                'violation_count': 0,
                'summary': 'Failed to generate statistics'
            }
    
    def _calculate_budget_utilization(self) -> float:
        """Calculate privacy budget utilization rate."""
        if not self.privacy_budget_tracking:
            return 0.0
        
        total_budget = sum(session.get('total_budget', 10.0) for session in self.privacy_budget_tracking.values())
        consumed_budget = sum(session.get('consumed_budget', 0.0) for session in self.privacy_budget_tracking.values())
        
        return consumed_budget / total_budget if total_budget > 0 else 0.0

