# Copyright 2026 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standard ordered preparation entry point for the curb-detection dataset.

Default: download and unpack the public curb-detection dataset (about 2 GB).
--smoke: generate a small synthetic dataset in the same layout, enough for
every lifelong-learning round to train and evaluate both tasks (real, sim).
It checks that the pipeline runs end to end; its accuracy is meaningless.
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://kubeedge.obs.cn-north-1.myhuaweicloud.com/ianvs/curb-detection/curb-detection.zip"

# Real samples are recognised by a Cityscapes city name in their path.
CITIES = ("aachen", "berlin", "bochum", "bremen", "cologne", "darmstadt")
NUM_LABEL_IDS = 19  # Cityscapes train ids 0..18; 255 is "ignore"
IMAGE_SIZE = (512, 256)


def _generate_split(split_dir: Path, samples_per_origin: int, rng) -> int:
    from PIL import Image

    images = split_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(samples_per_origin):
        city = CITIES[i % len(CITIES)]
        pairs = (
            (f"real_{city}_{i:06d}_000019_leftImg8bit.png",
             f"real_{city}_{i:06d}_000019_gtFine_labelTrainIds.png"),
            (f"sim_{i:04d}.png", f"sim_{i:04d}TrainIds.png"),
        )
        for image_name, label_name in pairs:  # interleave real and sim
            width, height = IMAGE_SIZE
            pixels = rng.integers(0, 256, (height, width, 3), dtype="uint8")
            Image.fromarray(pixels).save(images / image_name)
            labels = rng.integers(0, NUM_LABEL_IDS, (height, width), dtype="uint8")
            Image.fromarray(labels).save(images / label_name)
            lines.append(f"./images/{image_name} ./images/{label_name}\n")
    (split_dir / "index.txt").write_text("".join(lines), encoding="utf-8")
    return len(lines)


def _download(dataset_root: Path) -> None:
    archive = dataset_root.parent / "curb-detection.zip"
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_URL} (about 2 GB) ...")
    urllib.request.urlretrieve(DATASET_URL, archive)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(dataset_root.parent)
    archive.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the curb-detection dataset in Ianvs example layout."
    )
    parser.add_argument(
        "--dataset-root",
        "--output-dir",
        dest="dataset_root",
        type=Path,
        default=Path("./dataset/curb-detection"),
        help="Destination dataset root. Defaults to ./dataset/curb-detection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing files.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate a small synthetic dataset instead of downloading.",
    )
    parser.add_argument(
        "--samples-per-origin",
        type=int,
        default=8,
        help="Smoke mode: training samples per origin (real, sim). Default 8.",
    )
    args = parser.parse_args()

    train_index = args.dataset_root / "train_data" / "index.txt"
    if train_index.exists() and not args.overwrite:
        print(f"{train_index} already exists; use --overwrite to replace it.", file=sys.stderr)
        return 1

    if not args.smoke:
        _download(args.dataset_root)
        print(f"Prepared {args.dataset_root}")
        return 0

    import numpy as np

    rng = np.random.default_rng(0)
    n_train = _generate_split(args.dataset_root / "train_data", args.samples_per_origin, rng)
    n_test = _generate_split(args.dataset_root / "test_data",
                             max(1, args.samples_per_origin // 4), rng)
    print(f"Prepared smoke dataset in {args.dataset_root}: {n_train} train, {n_test} test samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
