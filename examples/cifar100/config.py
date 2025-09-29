import os
import glob
from core.common.log import LOGGER

# Choose between 'absolute' and 'relative' path modes
PATH_MODE = "relative"

USER_DEFINED_ABS_PATH = "/home/username/custom_path_here"
BASE_PATH = USER_DEFINED_ABS_PATH if PATH_MODE == "absolute" else os.path.dirname(os.path.abspath(__file__))

# Replace with your actual example name
EXAMPLE_NAME = "federated_learning/fedavg"

# Input and output directories
DATA_ROOT = os.path.join(BASE_PATH, "data")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
OUTPUT_ROOT = os.path.join(BASE_PATH, EXAMPLE_NAME, "output")

# Replace with your actual model name
MODEL_DIR = os.path.join(BASE_PATH,"init_model/restnet.pb")

# YAML configuration files
YAML_DIR = os.path.join(BASE_PATH, EXAMPLE_NAME)
YAML_FILES = glob.glob(os.path.join(YAML_DIR, "**", "*.yaml"), recursive=True)

# Dataset index files
TRAIN_INDEX_FILE = os.path.join(DATA_ROOT, "cifar100_train.txt")
TEST_INDEX_FILE = os.path.join(DATA_ROOT, "cifar100_test.txt")


def update_yaml_placeholders(files=None):
    files = files or YAML_FILES
    placeholders = {
        "{{TRAIN_INDEX_FILE}}": TRAIN_INDEX_FILE,
        "{{TEST_INDEX_FILE}}": TEST_INDEX_FILE,
        "{{OUTPUT_DIR}}": OUTPUT_ROOT,
        "{{MODEL_DIR}}": MODEL_DIR,
    }
    for fpath in files:
        with open(fpath) as f:
            content = f.read()
        for ph, val in placeholders.items():
            content = content.replace(ph, val)
        with open(fpath, "w") as f:
            f.write(content)
        LOGGER.info("Updated YAML: %s", fpath)


if __name__ == "__main__":
    LOGGER.info("Initial Model path: %s", MODEL_DIR)
    LOGGER.info("Train: %s", TRAIN_DIR)
    LOGGER.info("Test: %s", TEST_DIR)
    LOGGER.info("Output: %s", OUTPUT_ROOT)
    LOGGER.info("YAMLs: %s", YAML_FILES)
    update_yaml_placeholders()
