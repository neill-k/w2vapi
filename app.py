import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from gensim.models import KeyedVectors
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GloVe Word Embeddings API",
    description="API for serving GloVe word embeddings based on Wikipedia and Gigaword dataset",
    version="1.0.0",
)

# Load the model at startup
try:
    logger.info("Loading GloVe model...")
    import os
    model_path = "glove-wiki-gigaword-300.model"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}. Make sure to run 'git lfs pull'")
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure Git LFS files are properly downloaded.")
    
    model = KeyedVectors.load(model_path)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e!s}")
    raise


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
    try:
        # Try to access a common word to verify model is working
        vector = model["test"].tolist()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "model_loaded": False, "error": str(e)}


@app.post("/embedding", response_model=WordEmbedding)
async def get_embedding(word_input: WordInput):
    """Get the embedding vector for a single word."""
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
    try:
        similar_words = model.most_similar(word, topn=n)
        return {"similar_words": [{"word": word, "similarity": float(score)} for word, score in similar_words]}
    except KeyError as err:
        raise HTTPException(status_code=404, detail=f"Word '{word}' not found in vocabulary") from err
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
