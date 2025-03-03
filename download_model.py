
"""
Script to download the GloVe model during build phase
"""
import logging
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, login

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    # Create cache directory for models
    CACHE_DIR = Path("model_cache")
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Use HF token if available for faster downloads
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        logger.info("Using Hugging Face token for authentication")
        login(token=token)
    
    logger.info("Downloading GloVe model during build phase...")
    try:
        # Download both model files
        model_file = hf_hub_download(
            repo_id="fse/glove-wiki-gigaword-300",
            filename="glove-wiki-gigaword-300.model",
            local_dir=CACHE_DIR,
            token=token
        )
        logger.info(f"Downloaded model file to: {model_file}")
        
        vectors_file = hf_hub_download(
            repo_id="fse/glove-wiki-gigaword-300",
            filename="glove-wiki-gigaword-300.model.vectors.npy",
            local_dir=CACHE_DIR,
            token=token
        )
        logger.info(f"Downloaded vectors file to: {vectors_file}")
        
        # Verify files exist and have content
        if os.path.exists(model_file) and os.path.getsize(model_file) > 0 and \
           os.path.exists(vectors_file) and os.path.getsize(vectors_file) > 0:
            logger.info("Model downloaded successfully!")
            return True
        else:
            logger.error("Files downloaded but appear to be empty or corrupted")
            return False
    except Exception as e:
        logger.error(f"Error downloading model: {e!s}")
        return False

if __name__ == "__main__":
    success = download_model()
    # Exit with error code if download failed
    if not success:
        exit(1)
