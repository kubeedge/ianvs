<<<<<<< HEAD
import os
import argparse
from modelscope import snapshot_download
import logging

logging.getLogger().setLevel(logging.INFO)

def download_model(model_id, revision, local_dir):
    try:
        model_dir = snapshot_download(model_id, revision=revision, cache_dir=local_dir)
        logging.info(f"Model successfully downloaded to: {model_dir}")
        return model_dir
    except Exception as e:
        logging.info(f"Error downloading model: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from ModelScope")
    parser.add_argument("--model_id", type=str, required=True, help="ModelScope model ID")
    parser.add_argument("--revision", type=str, default="master", help="Model revision")
    parser.add_argument("--local_dir", type=str, required=True, help="Local directory to save the model")
    
    args = parser.parse_args()
    
    download_model(args.model_id, args.revision, args.local_dir)
=======
version https://git-lfs.github.com/spec/v1
oid sha256:5fed69345bdac448701f03dd7b3b41f82e7330ce4918df10f124e0b0f511b7d5
size 981
>>>>>>> 9676c3e (ya toh aar ya toh par)
