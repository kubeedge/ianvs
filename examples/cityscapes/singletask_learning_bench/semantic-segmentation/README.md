# Single Task Learning: Semantic Segmentation on Cityscapes

This example benchmarks RFNet-based semantic segmentation under Ianvs's `singletasklearning` paradigm, using the [Cityscapes](https://www.cityscapes-dataset.com/) urban street scene dataset.

## Ianvs Preparation

Follow the [Ianvs installation guide](../../../../docs/guides/how-to-install-ianvs.md) to install Ianvs before running this example.

## Dataset Preparation

Download the Cityscapes dataset from the [official downloads page](https://www.cityscapes-dataset.com/downloads/) (registration required). See [Semantic Segmentation: Cityscapes and SYNTHIA](../../../../docs/proposals/algorithms/lifelong-learning/Additional-documentation/curb_detetion_datasets.md) for a description of the dataset layout.

Place the dataset under `./dataset/cityscapes/` at the root of the Ianvs repo, and prepare `train_data.txt`/`test_data.txt` index files where each line lists an image path and its corresponding label path, separated by a space:

```
./images/aachen_000000_000019_leftImg8bit.png ./images/aachen_000000_000019_gtFine_labelTrainIds.png
```

## Model Preparation

`testalgorithms/rfnet/rfnet_algorithm.yaml` expects an initial RFNet model checkpoint at `./models/530_exp3_2.pth` (relative to the repo root). Place a pretrained checkpoint at that path, or update `initial_model_url` to point elsewhere.

## Run Ianvs

From the root directory of Ianvs:

```shell
ianvs -f examples/cityscapes/singletask_learning_bench/semantic-segmentation/benchmarkingjob.yaml
```

The leaderboard, ranked by the `map` metric, will be printed to the console and saved under `./workspace`.
