## Quick Start: Running the `federated_learning/fedavg` Example

Follow these steps to set up your environment and run the standard Federated Averaging (FedAvg) example.

### 1. Set Up the Environment

Ensure you have Python 3.10 or above installed to run this example.

Make sure Ianvs is installed on your system by following the instructions in the root `README.md`. Verify that all required dependencies are installed.

From the root `ianvs` directory, run the following:

```bash
# Step 1: Navigate to the root ianvs folder
cd ianvs

# Step 2: Create a Python virtual environment
python -m venv venv

# Step 3: Activate the virtual environment
source venv/bin/activate

# Step 4: Install root project dependencies
pip install -r requirements.txt

# Step 5: Install any additional libraries needed for this example
pip install -r examples/cifar100/federated_learning/fedavg/requirements.txt
```

### 2. Configure Example Paths

Locate the `examples/cifar100/config.py` file. This file contains all input, output, model, example\_name, and YAML file paths. Ensure the `EXAMPLE_NAME` variable is set to `"federated_learning/fedavg"`.

Run the configuration script from the project root:

```bash
python examples/cifar100/config.py
```

This will automatically update all `*.yaml` files in the subdirectories with the correct local paths for your machine. You should see output confirming that the YAML files have been updated.

### 3. Prepare the CIFAR-100 Dataset

Use the provided utility script to download and process the CIFAR-100 dataset into the format required by Ianvs. This will create a `data/` directory under `examples/cifar100`:

```bash
python examples/cifar100/utils.py
```

### 4. Run the Benchmarking Job

You are now ready to run the FedAvg example. Ianvs is launched via its main entry point, taking the `benchmarkingjob.yaml` as input. From the project root, run:

```bash
ianvs -f ./examples/cifar100/federated_learning/fedavg/benchmarkingjob.yaml
```

This will start the federated learning process as defined in the configuration files, training the model on the CIFAR-100 dataset and saving the output to `examples/cifar100/federated_learning/fedavg/output`.