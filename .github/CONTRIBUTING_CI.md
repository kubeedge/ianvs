# CI/CD Contributor Guide

## For Contributors

### Pre-Submission Checklist

Before submitting a pull request, complete these steps:

1. Test your changes locally:
```bash
   cd ~/projects/ianvs
   ./.github/scripts/test_locally.sh
```

2. Verify code quality:
```bash
   flake8 your_changed_files.py
```

3. Test module imports:
```bash
   python -c "from your_module import YourClass"
```

4. Follow PR title format:
```
   type(scope): description
   
   Valid types: feat, fix, docs, refactor, test, chore, ci
   
   Examples:
   feat(lifelong-learning): add new metric validation
   fix(ci): resolve dependency conflict  
   docs(workflow): update CI/CD documentation
```

### Understanding CI Checks

When you submit a pull request, several automated workflows execute:

#### PR Validation (approximately 5 minutes)
- Verifies PR title follows conventional commit format
- Validates presence of meaningful PR description
- Checks Python code quality using flake8
- Tests that changed modules can be imported
- Detects potential debug code

#### Example Validation (approximately 15-20 minutes per Python version)
- Tests on Python 3.8, 3.9, and 3.10
- Validates dependency compatibility
- Executes the example with reduced epochs
- Checks for NaN values in metrics
- Verifies no critical errors in execution

### Handling CI Failures

If automated checks fail:

1. Click on the failed check in your pull request
2. Review the execution logs to identify the specific error
3. Fix the issue in your local repository
4. Push the fix to your branch
5. CI will automatically re-run on the new commit

### Common Issues and Solutions

#### Dependency Conflict
```
Error: pip check failed
Conflicting dependencies detected
```
Solution: Update requirements.txt with compatible version specifications

#### Import Error
```
ModuleNotFoundError: No module named 'package_name'
```
Solution: Add the missing package to requirements.txt

#### NaN Metrics
```
Error: NaN values detected in output
```
Solution: Review model loading and prediction code in basemodel.py

## For Maintainers

### Monitoring Workflows

1. Navigate to the Actions tab in the repository
2. Review recent workflow executions
3. Identify any failing workflows
4. Download and review artifacts for detailed information

### Manual Workflow Execution

1. Go to the Actions tab
2. Select the desired workflow
3. Click "Run workflow"
4. Configure branch and any input parameters
5. Click "Run workflow" to start execution

### Workflow Maintenance

To modify a workflow:

1. Edit the workflow file in .github/workflows/
2. Validate YAML syntax:
```bash
   yamllint .github/workflows/*.yml
```
3. Create a draft pull request to test changes
4. Merge after successful validation

### Managing Persistent Failures

When a workflow consistently fails:

1. Review complete failure logs in the Actions tab
2. Check artifacts for detailed error information
3. Create an issue if the problem is recurring
4. Update documentation with the solution once resolved

## Workflow Schedule

- Example Validation: Triggered on every push and pull request
- Multi-Python Test: Executes daily at 2 AM UTC
- End-to-End Test: Runs weekly on Sunday at midnight UTC
- Failure Notifications: Triggered after workflow failures

## Getting Help

If you encounter issues with CI/CD:

- Consult .github/workflows/README.md for detailed workflow documentation
- Review workflow execution logs in the Actions tab
- Create an issue with the 'ci-help' label
- Contact the maintainers for persistent problems

## Resources

- GitHub Actions Documentation: https://docs.github.com/en/actions
- Workflow Syntax Reference: https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions
- Ianvs Project Documentation: https://github.com/kubeedge/ianvs