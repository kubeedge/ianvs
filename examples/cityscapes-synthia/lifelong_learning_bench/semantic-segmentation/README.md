# Cityscapes-SYNTHIA Semantic Segmentation — Lifelong Learning Benchmark

This benchmark evaluates lifelong learning algorithms on semantic segmentation using real
Cityscapes images and synthetic SYNTHIA images. The model is RFNet, trained incrementally
across rounds to handle both seen and unseen domain shifts.

## System Requirements

- OS: Ubuntu 18.04 or later (tested on Ubuntu 22.04)
- Python: 3.8 or later (tested on 3.12)
- CPU: 2 cores minimum, 4+ recommended
- RAM: 8 GB minimum
- Disk: 20 GB free minimum (the dataset alone is ~48 GB; see the small dataset option below)
- GPU: optional — training runs on CPU if no CUDA-capable GPU is detected


## Step 1. Clone and Install Ianvs

```shell
git clone https://github.com/kubeedge/ianvs.git
cd ianvs
```

Create and activate a virtual environment (strongly recommended to avoid dependency conflicts):

```shell
python3 -m venv venv
source venv/bin/activate
```

Install system-level dependencies:

```shell
sudo apt-get update
sudo apt-get install libgl1-mesa-glx -y
```

Install ianvs and its dependencies:

```shell
python -m pip install --upgrade pip
python -m pip install ./examples/resources/third_party/*
python -m pip install -r requirements.txt
pip install -e .
```

## Step 2. Install Sedna

Clone sedna and install it as an editable package so patches take effect immediately:

```shell
git clone https://github.com/kubeedge/sedna.git sedna_src
pip install -e sedna_src/
```

Then apply the two required patches to the cloned source before running the benchmark.

**Patch 1 — register UpdateStrategyDefault** (see [troubleshooting](#valueError-cant-find-class-type-knowledge_management-class-name-updatestrategydefault)):

In `sedna_src/lib/sedna/algorithms/seen_task_learning/__init__.py` add:

```python
from . import task_update_decision
```

**Patch 2 — add inference_2, my_evaluate, my_inference** (see [troubleshooting](#attributeerror-lifelonglearning-object-has-no-attribute-my_evaluate)):

Add the three methods listed in that troubleshooting section to
`sedna_src/lib/sedna/core/lifelong_learning/lifelong_learning.py`.

Verify the installation:

```shell
python -c "import sedna; print(sedna.__version__)"
```

## Step 3. Dataset Preparation

### Option A — Full dataset (48 GB, recommended for real results)

```shell
mkdir -p ~/data
cd ~/data
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com/sedna-robo/semantic_segmentation_dataset.zip
unzip semantic_segmentation_dataset.zip
```

Copy the index files shipped with this example into the dataset directory:

```shell
cp examples/cityscapes-synthia/lifelong_learning_bench/semantic-segmentation/indexes/* \
   ~/data/semantic_segmentation_dataset/
```

Then update `testenv/testenv.yaml` to point at the full index files:

```yaml
dataset:
  train_index: "/home/<your-user>/data/semantic_segmentation_dataset/train-index-sort.txt"
  test_index:  "/home/<your-user>/data/semantic_segmentation_dataset/test-index-sort.txt"
```

### Option B — Small dataset (for pipeline verification only)

The repo ships `indexes/train-small-3.txt` and `indexes/test-small-3.txt` which reference
a small subset of the full dataset (Cityscapes real + SYNTHIA sim images). Copy these into
your dataset directory:

```shell
cp examples/cityscapes-synthia/lifelong_learning_bench/semantic-segmentation/indexes/train-small-3.txt \
   ~/data/semantic_segmentation_dataset/
cp examples/cityscapes-synthia/lifelong_learning_bench/semantic-segmentation/indexes/test-small-3.txt \
   ~/data/semantic_segmentation_dataset/
```

The `testenv/testenv.yaml` is pre-configured to use `train-small-3.txt` and `test-small-3.txt`.
You still need to download the corresponding image files from the full dataset (the small index
references real images — no synthetic placeholder images are used).

## Step 4. Update Configuration Paths

Open `testenv/testenv.yaml` and set the **absolute paths** to match your machine:

```yaml
dataset:
  train_index: "/home/<your-user>/data/semantic_segmentation_dataset/train-small-3.txt"
  test_index:  "/home/<your-user>/data/semantic_segmentation_dataset/test-small-3.txt"
```

Replace `<your-user>` with your actual Linux username.

The `benchmarkingjob.yaml` workspace uses a relative path (`./workspace/semantic-segmentation`)
and does not need to be changed.

## Step 5. Set PYTHONPATH

RFNet has internal imports (e.g. `from mypath import Path`) that require the `RFNet/`
directory itself on `PYTHONPATH`. Run this export from the ianvs repo root every time
you open a new terminal, or add it to your shell profile:

```shell
export RFNET_DIR="examples/cityscapes-synthia/lifelong_learning_bench/semantic-segmentation/testalgorithms/rfnet/RFNet"
export PYTHONPATH="$RFNET_DIR:$PYTHONPATH"
```

## Step 6. Run the Benchmark

Activate the virtual environment and run ianvs from the repo root:

```shell
source venv/bin/activate

export RFNET_DIR="examples/cityscapes-synthia/lifelong_learning_bench/semantic-segmentation/testalgorithms/rfnet/RFNet"
export PYTHONPATH="$RFNET_DIR:$PYTHONPATH"

rm -rf workspace/semantic-segmentation

ianvs -f examples/cityscapes-synthia/lifelong_learning_bench/semantic-segmentation/benchmarkingjob.yaml
```

The benchmark runs `incremental_rounds: 2` of training by default (configurable in
`testenv/testenv.yaml`). Each round trains RFNet for 50 epochs on the current round's data
slice. On CPU with the small dataset, a complete run takes approximately 15–30 minutes.

## Expected Output

At the end of the run, ianvs prints a leaderboard table to the console:

```
+------+-------------------------+-----------+------------------------+------------------+-----------+
| rank |        algorithm        |  accuracy | samples_transfer_ratio |     paradigm     | basemodel |
+------+-------------------------+-----------+------------------------+------------------+-----------+
|  1   | rfnet_lifelong_learning |   0.3009  |         0.4807         | lifelonglearning | BaseModel |
+------+-------------------------+-----------+------------------------+------------------+-----------+
```

> **Note:** Results depend heavily on the dataset size. With the small dataset
> (6 training samples), expect lower accuracy (~0.03) and `samples_transfer_ratio` of ~0.33.
> The numbers above are from a full-dataset run.

Full results are saved under `./workspace/semantic-segmentation/`.

## Troubleshooting

### FileNotFoundError on startup

**Symptom:** ianvs exits immediately with a path that does not exist.

**Cause:** The `testenv/testenv.yaml` dataset paths use absolute paths from the original
author's machine.

**Fix:** Update the absolute paths in `testenv/testenv.yaml` to match your system:

```yaml
dataset:
  train_index: "/home/<your-user>/data/semantic_segmentation_dataset/train-small-3.txt"
  test_index:  "/home/<your-user>/data/semantic_segmentation_dataset/test-small-3.txt"
```

### NotImplementedError or dataset not loading

**Symptom:** Training starts but the dataset is empty, or a `NotImplementedError` is raised
during dataset loading.

**Cause:** Using `train_url` / `test_url` as dataset keys in `testenv.yaml`. Sedna's
`Dataset.process_dataset()` only recognises `train_index` / `test_index`.

**Fix:** In `testenv/testenv.yaml`, rename the keys:

```yaml
# wrong
train_url: "/path/to/train.txt"
test_url:  "/path/to/test.txt"

# correct
train_index: "/path/to/train.txt"
test_index:  "/path/to/test.txt"
```

### RuntimeError: No CUDA GPUs are available

**Symptom:** Crash at the start of training before any epoch runs.

**Cause:** `RFNet/utils/args.py` originally hardcoded `cuda=True` regardless of whether a
GPU is present.

**Fix:** This is already patched in this repository. If you see this error, verify that
`RFNet/utils/args.py` uses `torch.cuda.is_available()` instead of `True`:

```python
self.no_cuda = not torch.cuda.is_available()
self.cuda    = torch.cuda.is_available()
```

### DataLoader worker deadlock (process hangs with no output)

**Symptom:** The benchmark hangs silently after the first training log line, with CPU usage
stuck at a low value.

**Cause:** `DataLoader(num_workers=4)` uses `fork`-based multiprocessing on Linux. When
spawned from inside the ianvs process, the forked workers deadlock on CUDA context locks or
shared semaphores.

**Fix:** This is already patched in this repository (`workers=0` in `args.py`). If you
re-introduce `workers > 0`, add `multiprocessing.set_start_method('spawn')` at the top of
`basemodel.py`.

### TypeError: make_grid() got an unexpected keyword argument 'range'

**Symptom:** Crash during training with a `TypeError` pointing to `summaries.py`.

**Cause:** `torchvision` renamed the `range=` parameter to `value_range=` in version 0.9.0.

**Fix:** This is already patched in this repository. If you are using a custom copy of
RFNet, replace all occurrences in `RFNet/utils/summaries.py`:

```python
# old
make_grid(..., normalize=False, range=(0, 255))

# new
make_grid(..., normalize=False, value_range=(0, 255))
```

### OSError: [Errno 28] No space left on device

**Symptom:** Crash during model save at the end of a training round, with an error referencing
`/tmp/cityscapes/RFNet/experiment_N/checkpoint_*.pth`.

**Cause:** The original `Saver` class created a new `experiment_N` directory on every run
and never deleted old checkpoints, accumulating several gigabytes of `.pth` files in `/tmp`.

**Fix:** This is already patched in this repository. `Saver.__init__` now deletes all previous
experiment directories on startup, and `save_checkpoint` deletes the previous epoch's
checkpoint before writing the new one, keeping at most one `.pth` file on disk at any time.

To manually free space from a previous run:

```shell
rm -rf /tmp/cityscapes/
```

To redirect checkpoints away from `/tmp` (for example, to a partition with more space):

```shell
export TMPDIR=/path/to/large/disk
```

### ValueError: can't find class type knowledge_management class name UpdateStrategyDefault

**Symptom:** Crash at the start of round 1 with a `ValueError` from the ClassFactory registry.

**Cause:** The `task_update_decision` subpackage was never imported in
`sedna/algorithms/seen_task_learning/__init__.py`, so the `@ClassFactory.register` decorators
on `UpdateStrategyDefault` and `UpdateStrategyByFinetune` never executed and neither class
was registered.

**Fix:** In your sedna installation, open
`sedna_src/lib/sedna/algorithms/seen_task_learning/__init__.py` and add:

```python
from . import task_update_decision
```

If you installed sedna as an editable package (`pip install -e sedna_src/`) the change takes
effect immediately without reinstalling.

### AttributeError: 'LifelongLearning' object has no attribute 'my_evaluate'

**Symptom:** Crash during the evaluation phase with `AttributeError: 'LifelongLearning'
object has no attribute 'my_evaluate'` or `'my_inference'` or `'inference_2'`.

**Cause:** The ianvs paradigm layer calls `job.my_evaluate(...)`, `job.my_inference(...)`,
and `job.inference_2(...)` on the sedna `LifelongLearning` object, but these methods are not
present in the upstream sedna release.

**Fix:** Add the following three methods to the `LifelongLearning` class in your sedna
installation at `sedna_src/lib/sedna/core/lifelong_learning/lifelong_learning.py`:

```python
def inference_2(self, data=None, post_process=None, **kwargs):
    """Simplified inference for local/standalone testing."""
    from sedna.algorithms.unseen_task_detection.unseen_sample_recognition.\
        unseen_sample_recognition import SampleRegonitionDefault
    task_index_url = Context.get_parameters(
        "MODEL_URLS", self.cloud_knowledge_management.task_index)
    index_url = self.cloud_knowledge_management.local_task_index_url
    if not os.path.exists(index_url):
        FileOps.download(task_index_url, index_url)
    try:
        recognizer = SampleRegonitionDefault(index_url)
        _, unseen_samples = recognizer(data)
        is_unseen_task = len(unseen_samples.x) > 0
    except Exception:
        is_unseen_task = False
    res, tasks = self.cloud_knowledge_management.seen_estimator.predict(
        data=data, task_index=index_url, task_type="seen_task", **kwargs)
    return res, is_unseen_task, tasks

def my_evaluate(self, data, post_process=None, **kwargs):
    task_index_url = Context.get_parameters(
        "MODEL_URLS", self.cloud_knowledge_management.task_index)
    index_url = self.cloud_knowledge_management.local_task_index_url
    if not os.path.exists(index_url):
        FileOps.download(task_index_url, index_url)
    res, tasks_detail = self.cloud_knowledge_management.seen_estimator.evaluate(
        data=data, task_index=index_url, **kwargs)
    return index_url, tasks_detail, res

def my_inference(self, data=None, **kwargs):
    task_index_url = Context.get_parameters(
        "MODEL_URLS", self.cloud_knowledge_management.task_index)
    index_url = self.cloud_knowledge_management.local_task_index_url
    if not os.path.exists(index_url):
        FileOps.download(task_index_url, index_url)
    res, _ = self.cloud_knowledge_management.seen_estimator.predict(
        data=data, task_index=index_url, task_type="seen_task", **kwargs)
    return res
```

### samples_transfer_ratio is always 0.0

**Symptom:** The benchmark completes successfully but `samples_transfer_ratio` in the
leaderboard is `0.0`.

**Cause:** The dataset has too few inference samples and the per-sample unseen detection
uses a random coin flip — with only 2–3 samples, all can randomly be classified as "seen".

**Fix:** This is already handled in this repository by running the unseen-sample recognizer
on the full inference batch rather than sample-by-sample. If you still see ratio = 0,
verify that `core/testcasecontroller/algorithm/paradigm/lifelong_learning/lifelong_learning.py`
calls `SampleRegonitionDefault` on the full `inference_dataset` before the per-sample loop,
not inside it.

## Further Reading

- [Ianvs contributing guide](https://github.com/kubeedge/ianvs/blob/main/CONTRIBUTING.md)
- [How to test algorithms](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html)
- [Sedna lifelong learning API](https://github.com/kubeedge/sedna)
- [Issue tracker](https://github.com/kubeedge/ianvs/issues)
