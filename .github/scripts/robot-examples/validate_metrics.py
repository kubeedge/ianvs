#!/usr/bin/env python3
"""
Metrics validation for CI/CD pipeline
"""
import sys
import os
import glob
import re
import json
from pathlib import Path

class MetricsValidator:
    def __init__(self, workspace_dir="workspace"):
        self.workspace_dir = workspace_dir
        self.errors = []
        self.warnings = []
        self.metrics = {}
    
    def validate_all(self):
        """Execute all validation checks"""
        print("Starting metrics validation...\n")
        
        if not self.check_output_exists():
            self.warnings.append("Output files not found - may be first run")
        
        if not self.check_no_nan():
            self.errors.append("NaN values detected in metrics")
        
        if not self.check_metric_ranges():
            self.warnings.append("Metrics validation incomplete")
        
        return len(self.errors) == 0
    
    def check_output_exists(self):
        """Verify output files were generated"""
        print("Checking output files...")
        
        if not os.path.exists(self.workspace_dir):
            print(f"Workspace directory not found: {self.workspace_dir}")
            return False
        
        log_files = glob.glob(f"{self.workspace_dir}/**/*.log", recursive=True)
        if not log_files:
            print("No log files found")
            return False
        
        print(f"Found {len(log_files)} log file(s)")
        return True
    
    def check_no_nan(self):
        """Check for NaN values"""
        print("\nChecking for NaN values...")
        
        found_nan = False
        
        for log_file in glob.glob(f"{self.workspace_dir}/**/*.log", recursive=True):
            with open(log_file, 'r') as f:
                content = f.read()
                if re.search(r'\bnan\b', content, re.IGNORECASE):
                    print(f"ERROR: NaN found in {log_file}")
                    found_nan = True
        
        if not found_nan:
            print("No NaN values detected")
            return True
        return False
    
    def check_metric_ranges(self):
        """Check metric ranges"""
        print("\nChecking metric ranges...")
        
        expected_ranges = {
            'mIoU': (0.0, 1.0),
            'CPA': (0.0, 1.0),
            'accuracy': (0.0, 1.0)
        }
        
        for log_file in glob.glob(f"{self.workspace_dir}/**/*.log", recursive=True):
            with open(log_file, 'r') as f:
                content = f.read()
                
                for metric, (min_val, max_val) in expected_ranges.items():
                    pattern = rf'{metric}[:\s]+([0-9.]+)'
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        try:
                            value = float(match)
                            self.metrics[metric] = value
                            
                            if min_val <= value <= max_val:
                                print(f"{metric}: {value:.4f} (valid)")
                            else:
                                print(f"WARNING: {metric}: {value:.4f} out of range")
                        except ValueError:
                            pass
        
        if self.metrics:
            return True
        
        print("No metrics found yet")
        return False
    
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        
        if self.metrics:
            print("\nMetrics:")
            for metric, value in self.metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors:
            print("\nValidation passed")
        
        report = {
            'metrics': self.metrics,
            'errors': self.errors,
            'warnings': self.warnings,
            'passed': len(self.errors) == 0
        }
        
        with open('validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nReport saved to validation_report.json")

if __name__ == "__main__":
    validator = MetricsValidator()
    
    success = validator.validate_all()
    validator.generate_report()
    
    sys.exit(0 if success else 1)
