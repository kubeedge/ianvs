## Quick Start: Running the `federated_learning/fedavg` Example

Follow these steps to set up your environment and run the standard Federated Averaging (FedAvg) example.

### 1. Set Up the Environment

Ensure you have Python 3.10 or above installed to run this example.

Make sure Ianvs is installed on your system by following the instructions in the root `README.md`. Verify that all required dependencies are installed.

Next, navigate to the `examples/cifar100/federated_learning/fedavg` directory and install the dependencies specific to this example:

```bash
# Step 1: Navigate to the root ianvs folder
cd ianvs

# Step 2: Create a Python virtual environment
python -m venv venv

# Step 3: Activate the virtual environment
source venv/bin/activate

# Step 4: Install root project dependencies
pip install -r requirements.txt

# Step 5: Navigate to the CIFAR-100 examples directory
cd examples/cifar100/federated_learning/fedavg

# Step 6: Install any additional libraries needed for this example
# If you face dependency conflicts, create a separate virtual environment,
# install the root requirements, and then example requirements.
pip install -r requirements.txt

```

### 2. Configure Example Paths

Locate the config.py file. This file contains all input, output, model, example_name, and YAML file paths. First, set the EXAMPLE_NAME variable to "federated_learning/fedavg". You can then edit any other paths in the configuration if needed.

Running `config.py` will automatically update all `*.yaml` files in the subdirectories with the correct local paths for your machine:

```bash
python config.py
```

You should see output indicating that the YAML files have been successfully updated.

### 3. Prepare the CIFAR-100 Dataset

Use the provided utility script to download and process the CIFAR-100 dataset into the format required by Ianvs. This will create a `data/` directory with the prepared dataset:

```bash
python utils.py
```

### 4. Run the Benchmarking Job

You are now ready to run the FedAvg example. Ianvs is typically launched via a main entry point that takes the `benchmarkingjob.yaml` as input.

Assuming the Ianvs runner is in your `PATH` or located at the root of the project, run:

```bash
# Run the federated learning benchmark
ianvs -f ./examples/cifar100/federated_learning/fedavg/benchmarkingjob.yaml
```

This will start the federated learning process as defined in the configuration files, training the model on the CIFAR-100 dataset and saving the output to `federated_learning/fedavg/output`.
