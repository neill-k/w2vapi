import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GloVe Word Embeddings API",
    description="API for serving GloVe word embeddings based on Wikipedia and Gigaword dataset",
    version="1.0.0",
)

# Create cache directory for models
CACHE_DIR = Path("model_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Initialize model as None
model = None

# We'll load the model in the startup event
@app.on_event("startup")
async def load_model():
    global model
    try:
        logger.info("========== STARTUP BEGIN ==========")
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Cache directory exists: {CACHE_DIR.exists()}")
        logger.info(f"Cache directory content: {list(CACHE_DIR.glob('*'))}")
        
        logger.info("Loading GloVe model...")
        logger.info("Attempting to load GloVe model...")
        model_path = CACHE_DIR / "glove-wiki-gigaword-300.model"
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model path exists: {model_path.exists()}")
        
        # First check the standard path
        if not model_path.exists():
            # Try alternative paths
            logger.warning("Standard model path not found, trying alternative paths")
            
            # Try with absolute path from current directory
            alt_paths = [
                Path(os.getcwd()) / "model_cache" / "glove-wiki-gigaword-300.model",
                Path("/home/runner/workspace/model_cache/glove-wiki-gigaword-300.model"),
                Path("./model_cache/glove-wiki-gigaword-300.model")
            ]
            
            # Check if the model exists in the cache directory directly
            cache_files = list(CACHE_DIR.glob("*")) if CACHE_DIR.exists() else []
            logger.info(f"Files in cache directory: {cache_files}")
            
            # Try each alternative path
            for alt_path in alt_paths:
                logger.info(f"Trying alternative path: {alt_path}")
                if alt_path.exists():
                    logger.info(f"Found model at alternative path: {alt_path}")
                    model_path = alt_path
                    break
            
            # If still not found, try to download
            if not model_path.exists():
                logger.warning("Model not found in any location, attempting to download")
                try:
                    from download_model import download_model
                    download_success = download_model()
                    if download_success:
                        logger.info("Model downloaded successfully")
                    else:
                        logger.error("Failed to download model")
                except Exception as e:
                    logger.error(f"Error downloading model: {e}")
            
            # Final check if model exists
            if not model_path.exists():
                error_msg = "Model files not found after all attempts. The build command should have downloaded them."
                logger.error(error_msg)
                logger.error("This likely means the build command failed or did not complete.")
                logger.error(f"Current directory: {os.getcwd()}")
                logger.error(f"All files in current directory: {os.listdir('.')}")
                logger.error(f"All files in model_cache: {os.listdir(CACHE_DIR) if CACHE_DIR.exists() else 'cache dir not found'}")
                raise RuntimeError(error_msg)
        
        logger.info(f"Loading model from: {model_path}")
        model = KeyedVectors.load(str(model_path))
        logger.info("Model loaded into memory")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model vocabulary size: {len(model.key_to_index)}")
        logger.info(f"Sample words in vocabulary: {list(model.key_to_index.keys())[:5]}")
        logger.info("Model loaded successfully!")
        logger.info("========== STARTUP COMPLETE ==========")
    except Exception as e:
        logger.error(f"========== ERROR LOADING MODEL ==========")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Error details: {repr(e)}")
        if model_path.exists():
            logger.error(f"Model file exists but loading failed")
            logger.error(f"Model file size: {os.path.getsize(model_path)}")
        else:
            logger.error(f"Model file does not exist")
        logger.error(f"Current directory files: {os.listdir('.')}")
        logger.error(f"Cache directory files: {os.listdir(CACHE_DIR) if CACHE_DIR.exists() else 'cache dir not found'}")
        logger.error("========== ERROR DETAILS END ==========")
        logger.error(f"Error loading model: {e!s}")
        # Don't raise the exception - let the API start without the model
        # We'll handle the None model in the endpoints


class WordInput(BaseModel):
    word: str


class WordsInput(BaseModel):
    words: list[str]


class WordEmbedding(BaseModel):
    embedding: list[float]


class WordEmbeddingResponse(BaseModel):
    embedding: Optional[list[float]]


class WordEmbeddingsResponse(BaseModel):
    results: dict[str, WordEmbeddingResponse]


class SimilarWord(BaseModel):
    word: str
    similarity: float


class SimilarWordsResponse(BaseModel):
    similar_words: list[SimilarWord]


@app.get("/")
async def root():
    return {"message": "GloVe Word Embeddings API", "model": "glove-wiki-gigaword-300", "dimensions": 300, "status": "running"}

@app.get("/health")
async def health_check():
    """Check if the model is loaded properly."""
    logger.info(f"Health check requested. Model is: {'loaded' if model is not None else 'None'}")
    
    if model is None:
        logger.info("Health check: Model is still initializing")
        return {"status": "initializing", "model_loaded": False, "message": "Model is still loading"}
    
    # Quick health check - just verify model exists without testing vectors
    # This makes health checks faster for deployment
    return {"status": "healthy", "model_loaded": True}


@app.post("/embedding", response_model=WordEmbedding)
async def get_embedding(word_input: WordInput):
    """Get the embedding vector for a single word."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still initializing, please try again later")
    
    try:
        vector = model[word_input.word].tolist()
        return {"embedding": vector}
    except KeyError as err:
        raise HTTPException(status_code=404, detail=f"Word '{word_input.word}' not found in vocabulary") from err
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/embeddings", response_model=WordEmbeddingsResponse)
async def get_embeddings(words_input: WordsInput):
    """Get embedding vectors for multiple words."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still initializing, please try again later")
    
    results = {}
    for word in words_input.words:
        try:
            vector = model[word].tolist()
            results[word] = {"embedding": vector}
        except KeyError:
            results[word] = {"embedding": None}
        except Exception:
            results[word] = {"embedding": None}
    return {"results": results}


@app.get("/similar/{word}", response_model=SimilarWordsResponse)
async def get_similar_words(word: str, n: int = 10):
    """Get n most similar words for a given word."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still initializing, please try again later")
    
    try:
        similar_words = model.most_similar(word, topn=n)
        return {"similar_words": [{"word": word, "similarity": float(score)} for word, score in similar_words]}
    except KeyError as err:
        raise HTTPException(status_code=404, detail=f"Word '{word}' not found in vocabulary") from err
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
