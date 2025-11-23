<<<<<<< HEAD
import os
import argparse
import logging
from modelscope import snapshot_download

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
oid sha256:e2b3f2e23340be21feb3af761a72de434cbd54265b8819f482d80fb614662de2
size 981
>>>>>>> 9676c3e (ya toh aar ya toh par)
