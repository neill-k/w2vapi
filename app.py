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
        print("========== STARTUP BEGIN ==========")
        print(f"Current directory: {os.getcwd()}")
        print(f"Cache directory exists: {CACHE_DIR.exists()}")
        print(f"Cache directory content: {list(CACHE_DIR.glob('*'))}")
        
        logger.info("Loading GloVe model...")
        print("Attempting to load GloVe model...")
        model_path = CACHE_DIR / "glove-wiki-gigaword-300.model"
        print(f"Model path: {model_path}")
        print(f"Model path exists: {model_path.exists()}")
        
        if not model_path.exists():
            error_msg = "Model files not found. The build command should have downloaded them."
            logger.error(error_msg)
            print(error_msg)
            print("This likely means the build command failed or did not complete.")
            raise RuntimeError(error_msg)
        
        print(f"Loading model from: {model_path}")
        model = KeyedVectors.load(str(model_path))
        print("Model loaded into memory")
        print(f"Model type: {type(model)}")
        print(f"Model vocabulary size: {len(model.key_to_index)}")
        print(f"Sample words in vocabulary: {list(model.key_to_index.keys())[:5]}")
        logger.info("Model loaded successfully!")
        print("========== STARTUP COMPLETE ==========")
    except Exception as e:
        print(f"========== ERROR LOADING MODEL ==========")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        print(f"Error details: {repr(e)}")
        if model_path.exists():
            print(f"Model file exists but loading failed")
            print(f"Model file size: {os.path.getsize(model_path)}")
        else:
            print(f"Model file does not exist")
        print(f"Current directory files: {os.listdir('.')}")
        print(f"Cache directory files: {os.listdir(CACHE_DIR) if CACHE_DIR.exists() else 'cache dir not found'}")
        print("========== ERROR DETAILS END ==========")
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
    print(f"Health check requested. Model is: {'loaded' if model is not None else 'None'}")
    
    if model is None:
        print("Health check: Model is still initializing")
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
