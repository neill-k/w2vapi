
"""
Script to download the GloVe model during build phase
"""
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    # Create cache directory for models
    CACHE_DIR = Path("model_cache")
    CACHE_DIR.mkdir(exist_ok=True)
    
    logger.info("Downloading GloVe model during build phase...")
    try:
        # Download both model files
        hf_hub_download(
            repo_id="fse/glove-wiki-gigaword-300",
            filename="glove-wiki-gigaword-300.model",
            local_dir=CACHE_DIR
        )
        hf_hub_download(
            repo_id="fse/glove-wiki-gigaword-300",
            filename="glove-wiki-gigaword-300.model.vectors.npy",
            local_dir=CACHE_DIR
        )
        logger.info("Model downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e!s}")
        return False

if __name__ == "__main__":
    download_model()
