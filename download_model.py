
"""
Script to download the GloVe model during build phase
"""
import logging
import os
import time
from pathlib import Path
from huggingface_hub import hf_hub_download, login

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    # Create cache directory for models
    CACHE_DIR = Path("model_cache")
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Log current directory and cache directory for debugging
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Cache directory absolute path: {CACHE_DIR.absolute()}")
    
    # Use HF token if available for faster downloads
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        logger.info("Using Hugging Face token for authentication")
        login(token=token)
    
    logger.info("Downloading GloVe model during build phase...")
    try:
        # Download both model files with retry logic
        for attempt in range(3):  # Try up to 3 times
            try:
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
                break  # If successful, exit retry loop
            except Exception as download_err:
                logger.warning(f"Download attempt {attempt+1} failed: {download_err}")
                if attempt == 2:  # Last attempt
                    raise  # Re-raise the exception if this was the last attempt
                time.sleep(2)  # Wait before retrying
        
        # List all files in cache directory to verify
        files_in_cache = list(CACHE_DIR.glob("*"))
        logger.info(f"Files in cache directory after download: {files_in_cache}")
        
        # Force write sync to ensure files are committed to disk
        os.sync()
        
        # Verify files exist and have content
        model_path = CACHE_DIR / "glove-wiki-gigaword-300.model"
        vectors_path = CACHE_DIR / "glove-wiki-gigaword-300.model.vectors.npy"
        
        if model_path.exists() and model_path.stat().st_size > 0 and \
           vectors_path.exists() and vectors_path.stat().st_size > 0:
            logger.info(f"Model file size: {model_path.stat().st_size} bytes")
            logger.info(f"Vectors file size: {vectors_path.stat().st_size} bytes")
            logger.info("Model downloaded successfully!")
            return True
        else:
            logger.error("Files downloaded but appear to be empty or corrupted")
            logger.error(f"Model exists: {model_path.exists()}, Vectors exists: {vectors_path.exists()}")
            return False
    except Exception as e:
        logger.error(f"Error downloading model: {e!s}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    success = download_model()
    # Exit with error code if download failed
    if not success:
        exit(1)
