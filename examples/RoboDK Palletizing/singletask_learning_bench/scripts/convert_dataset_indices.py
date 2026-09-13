"""
Converts the absolute dataset paths in train_index.txt / test_index.txt into
relative paths, so ianvs's built-in path-resolution logic
(core/testenvmanager/dataset/dataset.py: _process_txt_index_file) can correctly
resolve them relative to the index file's own location on any machine.

The RoboDK Palletizing dataset, as currently published on Kaggle, ships with
absolute paths baked into its index files, e.g.:
    /root/ianvs/project/data/dataset/RoboDK_Palletizing_Dataset/images/train/...
This script converts those to relative paths:
    images/train/...
which ianvs then resolves relative to wherever the user places the dataset.

This script is idempotent: running it on already-relative index files is a
safe no-op.

Usage:
    python convert_dataset_indices.py <path-to-RoboDK_Palletizing_Dataset-folder>
"""
import sys
from pathlib import Path

OLD_PREFIX = "/root/ianvs/project/data/dataset/RoboDK_Palletizing_Dataset/"


def convert_index_file(file_path: Path) -> None:
    content = file_path.read_text(encoding="utf-8")
    if OLD_PREFIX not in content:
        print(f"Already portable, no changes needed: {file_path}")
        return
    updated = content.replace(OLD_PREFIX, "")
    file_path.write_text(updated, encoding="utf-8")
    print(f"Converted to relative paths: {file_path}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python convert_dataset_indices.py <path-to-RoboDK_Palletizing_Dataset-folder>")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1]).resolve()

    if not dataset_dir.is_dir():
        print(f"Error: {dataset_dir} is not a directory.")
        sys.exit(1)

    for name in ("train_index.txt", "test_index.txt"):
        index_file = dataset_dir / name
        if not index_file.exists():
            print(f"Warning: {index_file} not found, skipping.")
            continue
        convert_index_file(index_file)


if __name__ == "__main__":
    main()
