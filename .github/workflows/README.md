# CI/CD Pipeline Documentation

## Overview

This directory contains automated workflows for validating the Ianvs lifelong learning example across multiple Python versions and environments.

## Workflow Files

### 1. example_validation.yml - Primary Example Validation

Purpose: Validates that the lifelong learning example executes correctly across supported Python versions.

Triggers:
- Push to main branch
- Pull requests affecting example files
- Manual dispatch

Tests performed:
- Python 3.8, 3.9, 3.10 compatibility
- Dependency installation and conflict detection
- Example execution with reduced epochs
- Metric validation to prevent NaN values
- Error detection in execution logs

Expected duration: 15-20 minutes per Python version

### 2. multi_python_test.yml - Compatibility Matrix Testing

Purpose: Comprehensive compatibility testing across operating systems and Python versions.

Triggers:
- Daily schedule at 2 AM UTC
- Manual dispatch with configurable Python versions

Tests performed:
- Ubuntu 20.04 and 22.04 compatibility
- Python 3.8, 3.9, 3.10 verification
- Package import validation
- Dependency conflict detection

Expected duration: 10 minutes per matrix combination

### 3. pr_validation.yml - Pull Request Quality Checks

Purpose: Validates pull request format, code quality, and safety before merge.

Triggers:
- Pull request events (opened, synchronized, reopened)

Checks performed:
- PR title follows conventional commit format
- PR description is present and meaningful
- Python code quality via flake8
- Import safety for changed modules
- Detection of debug code and TODOs
- PR size assessment

Expected duration: 5 minutes

## Usage

### Running Workflows Manually

1. Navigate to the Actions tab in the GitHub repository
2. Select the desired workflow from the left sidebar
3. Click "Run workflow" button
4. Select the target branch
5. Configure any workflow-specific inputs if available
6. Click "Run workflow" to start execution

### Viewing Results

1. Go to the Actions tab
2. Click on the workflow run of interest
3. Review the summary page for overall status
4. Click on individual jobs to view detailed logs
5. Download artifacts from the summary page if needed

### Interpreting Results

- Green checkmark: All validations passed
- Red X mark: One or more validations failed
- Yellow circle: Workflow is currently running
- Gray dash: Workflow was skipped or cancelled

## Artifacts

Each workflow run generates artifacts that are retained for 7 days:

- results-python-X.X: Contains execution logs and workspace files
- dependency-report-pyX.X: JSON format dependency analysis
- compat-*: Compatibility test results across OS and Python versions
- pr-summary: Pull request validation summary

## Troubleshooting

### Workflow Fails with Dependency Errors

Check the dependency-report artifact for conflicting packages. Update requirements.txt with compatible version specifications. Test locally using .github/scripts/test_locally.sh before pushing changes.

### Workflow Times Out

Verify that the example execution time is reasonable. Consider reducing the number of training epochs in the CI configuration. Check for any blocking operations or infinite loops in the code.

### NaN Metrics Detected

Review the prediction code in basemodel.py to ensure models are loaded correctly. Verify that the validator has access to trained model weights. Check run.log artifact for detailed error messages.

## Maintenance

### Adding New Validation Steps

To add a new step to a workflow:
```yaml
- name: Descriptive Step Name
  run: |
    echo "Executing validation..."
    # Add your commands here
```

Place the step in the appropriate position within the workflow file.

### Modifying Python Version Matrix

Edit the strategy matrix section:
```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11']
```

### Adjusting Workflow Triggers

Modify the on section to change when workflows execute:
```yaml
on:
  push:
    branches: [ main, develop, feature/* ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
```

## Best Practices

1. Test workflow changes locally before committing when possible
2. Use caching to reduce workflow execution time
3. Set fail-fast to false in matrix strategies to see all failures
4. Always upload artifacts for debugging purposes
5. Use descriptive names for jobs and steps
6. Document any non-obvious workflow logic
7. Keep workflows focused on specific validation tasks

## Contributing

When modifying workflows:

1. Validate YAML syntax before committing
2. Test on a feature branch first
3. Document any changes in this README
4. Consider backward compatibility
5. Update related documentation

## Resources

- GitHub Actions Documentation: https://docs.github.com/en/actions
- Workflow Syntax Reference: https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions
- Ianvs Project: https://github.com/kubeedge/ianvs