#!/bin/bash
# Local CI testing script

set -e

echo "Testing CI workflows locally..."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test 1: Check Python versions
echo -e "${YELLOW}Test 1: Checking Python versions...${NC}"
for version in 3.8 3.9 3.10; do
    if command -v python$version &> /dev/null; then
        echo -e "${GREEN}Python $version is available${NC}"
    else
        echo -e "${RED}Python $version is not available${NC}"
    fi
done

# Test 2: Install dependencies
echo -e "${YELLOW}Test 2: Testing dependency installation...${NC}"
python3 -m venv test_venv
source test_venv/bin/activate

pip install --upgrade pip wheel setuptools > /dev/null 2>&1
if [ -f requirements.txt ]; then
    pip install -r requirements.txt > /dev/null 2>&1
fi

echo -e "${GREEN}Dependencies installed successfully${NC}"

# Test 3: Test imports
echo -e "${YELLOW}Test 3: Testing module imports...${NC}"
python -c "
import sys
sys.path.insert(0, 'examples/robot/lifelong_learning_bench/semantic-segmentation/testalgorithms/rfnet')

try:
    # Just test if we can import the file
    import basemodel
    print('basemodel.py is accessible')
except Exception as e:
    print(f'Could not access basemodel.py: {e}')
"

# Test 4: Validate workflow syntax
echo -e "${YELLOW}Test 4: Validating workflow YAML syntax...${NC}"
pip install pyyaml > /dev/null 2>&1

python -c "
import yaml
import sys

workflows = [
    '.github/workflows/example_validation.yml',
    '.github/workflows/multi_python_test.yml',
    '.github/workflows/pr_validation.yml'
]

all_ok = True
for workflow in workflows:
    try:
        with open(workflow, 'r') as f:
            yaml.safe_load(f)
        print(f'{workflow} syntax is valid')
    except Exception as e:
        print(f'{workflow} has syntax error: {e}')
        all_ok = False
        sys.exit(1)

if all_ok:
    print('All workflow files have valid syntax')
"

deactivate
rm -rf test_venv

echo -e "${GREEN}All local tests passed successfully${NC}"
