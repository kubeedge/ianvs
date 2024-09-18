import os
import argparse
from huggingface_hub import hf_hub_download

def download_model(repo_id, filename, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=local_dir
        )
        print(f"Model successfully downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID")
    parser.add_argument("--filename", type=str, default="*q8_0.gguf", help="Filename or pattern to download")
    parser.add_argument("--local_dir", type=str, required=True, help="Local directory to save the model")
    
    args = parser.parse_args()
    
    download_model(args.repo_id, args.filename, args.local_dir)