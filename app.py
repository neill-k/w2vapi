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
    description=
    "API for serving GloVe word embeddings based on Wikipedia and Gigaword dataset",
    version="1.0.0",
)

# Create cache directory for models
CACHE_DIR = Path("model_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Initialize model as None
model = None

# Flag to track if model is being loaded
is_model_loading = False


# We'll start loading the model in the startup event but won't block startup
@app.on_event("startup")
async def startup_event():
    global is_model_loading
    import asyncio

    logger.info("========== STARTUP BEGIN ==========")
    logger.info("API server starting up, port will be available immediately")
    logger.info(f"Model will be loaded in the background")

    # Start model loading in the background
    is_model_loading = True
    asyncio.create_task(load_model_background())


async def load_model_background():
    """Load the model in the background without blocking API startup"""
    global model, is_model_loading
    try:
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Cache directory exists: {CACHE_DIR.exists()}")
        logger.info(f"Cache directory content: {list(CACHE_DIR.glob('*'))}")

        logger.info("Loading GloVe model in background...")
        model_path = CACHE_DIR / "glove-wiki-gigaword-300.model"
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model path exists: {model_path.exists()}")

        # First check the standard path
        if not model_path.exists():
            # Try alternative paths
            logger.warning(
                "Standard model path not found, trying alternative paths")

            # Try with absolute path from current directory
            alt_paths = [
                Path(os.getcwd()) / "model_cache" /
                "glove-wiki-gigaword-300.model",
                Path(
                    "/home/runner/workspace/model_cache/glove-wiki-gigaword-300.model"
                ),
                Path("./model_cache/glove-wiki-gigaword-300.model")
            ]

            # Check if the model exists in the cache directory directly
            cache_files = list(
                CACHE_DIR.glob("*")) if CACHE_DIR.exists() else []
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
                logger.warning(
                    "Model not found in any location, attempting to download")
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
                logger.error(
                    "This likely means the build command failed or did not complete."
                )
                logger.error(f"Current directory: {os.getcwd()}")
                logger.error(
                    f"All files in current directory: {os.listdir('.')}")
                logger.error(
                    f"All files in model_cache: {os.listdir(CACHE_DIR) if CACHE_DIR.exists() else 'cache dir not found'}"
                )
                is_model_loading = False
                return  # Don't raise exception, just return

        logger.info(f"Loading model from: {model_path}")
        model = KeyedVectors.load(str(model_path))
        logger.info("Model loaded into memory")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model vocabulary size: {len(model.key_to_index)}")
        logger.info(
            f"Sample words in vocabulary: {list(model.key_to_index.keys())[:5]}"
        )
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
        logger.error(
            f"Cache directory files: {os.listdir(CACHE_DIR) if CACHE_DIR.exists() else 'cache dir not found'}"
        )
        logger.error("========== ERROR DETAILS END ==========")
        logger.error(f"Error loading model: {e!s}")
    finally:
        is_model_loading = False


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


class TokenizeInput(BaseModel):
    text: str
    model: str = "gpt-3.5-turbo"  #default model


class TokenizeResponse(BaseModel):
    tokens: list[int]
    token_count: int
    token_strings: list[str]


@app.get("/")
async def root():
    return {
        "message": "GloVe Word Embeddings API",
        "model": "glove-wiki-gigaword-300",
        "dimensions": 300,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Check if the model is loaded properly."""
    logger.info(
        f"Health check requested. Model is: {'loaded' if model is not None else 'None'}"
    )

    if is_model_loading:
        logger.info("Health check: Model is currently loading")
        return {
            "status": "initializing",
            "model_loaded": False,
            "message": "Model is currently loading in background"
        }
    elif model is None:
        logger.info(
            "Health check: Model failed to load or hasn't started loading yet")
        return {
            "status": "warning",
            "model_loaded": False,
            "message": "Model not loaded, but API is operational"
        }

    # Quick health check - just verify model exists without testing vectors
    # This makes health checks faster for deployment
    return {"status": "healthy", "model_loaded": True}


@app.post("/embedding", response_model=WordEmbedding)
async def get_embedding(word_input: WordInput):
    """Get the embedding vector for a single word."""
    if is_model_loading:
        raise HTTPException(
            status_code=503,
            detail=
            "Model is currently loading, please try again in a few moments")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=
            "Model is not loaded, please check health endpoint for status")

    try:
        vector = model[word_input.word.lower().strip()].tolist()
        return {"embedding": vector}
    except KeyError as err:
        raise HTTPException(
            status_code=404,
            detail=
            f"Word '{word_input.word.lower().strip()}' not found in vocabulary"
        ) from err
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/embeddings", response_model=WordEmbeddingsResponse)
async def get_embeddings(words_input: WordsInput):
    """Get embedding vectors for multiple words."""
    if is_model_loading:
        raise HTTPException(
            status_code=503,
            detail=
            "Model is currently loading, please try again in a few moments")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=
            "Model is not loaded, please check health endpoint for status")

    results = {}
    for word in words_input.words:
        normalized_word = word.lower().strip()
        try:
            vector = model[normalized_word].tolist()
            results[normalized_word] = {"embedding": vector}
        except KeyError:
            results[normalized_word] = {"embedding": None}
        except Exception:
            results[normalized_word] = {"embedding": None}
    return {"results": results}


@app.get("/similar/{word}", response_model=SimilarWordsResponse)
async def get_similar_words(word: str, n: int = 10):
    """Get n most similar words for a given word."""
    if is_model_loading:
        raise HTTPException(
            status_code=503,
            detail=
            "Model is currently loading, please try again in a few moments")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=
            "Model is not loaded, please check health endpoint for status")

    try:
        similar_words = model.most_similar(word, topn=n)
        return {
            "similar_words": [{
                "word": word,
                "similarity": float(score)
            } for word, score in similar_words]
        }
    except KeyError as err:
        raise HTTPException(
            status_code=404,
            detail=f"Word '{word}' not found in vocabulary") from err
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize_text(tokenize_input: TokenizeInput):
    """Tokenize text using OpenAI's tiktoken library."""
    try:
        import tiktoken

        # Get appropriate encoding based on model
        encoding = tiktoken.encoding_for_model(tokenize_input.model)

        tokens = encoding.encode(tokenize_input.text)

        # Decode tokens to see token strings
        token_strings = [
            encoding.decode_single_token_bytes(token).decode('utf-8',
                                                             errors='replace')
            for token in tokens
        ]

        return {
            "tokens": tokens,
            "token_count": len(tokens),
            "token_strings": token_strings
        }
    except ModuleNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=
            "Tiktoken module not installed. Please add tiktoken to requirements.txt and restart the server."
        )
    except Exception as e:
        if "is not a supported model" in str(e):
            raise HTTPException(
                status_code=400,
                detail=
                f"Invalid model name: {tokenize_input.model}. Try 'gpt-3.5-turbo', 'gpt-4', or 'text-davinci-003'"
            )
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/available-tokenizers")
async def get_available_tokenizers():
    """Get a list of available tokenizers from tiktoken."""
    try:
        import tiktoken
        return {
            "available_models": [
                "gpt-4", "gpt-3.5-turbo", "text-davinci-003",
                "text-davinci-002", "text-davinci-001", "text-curie-001",
                "text-babbage-001", "text-ada-001", "davinci", "curie",
                "babbage", "ada"
            ],
            "default_encoding":
            tiktoken.get_encoding("cl100k_base").name
        }
    except ModuleNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=
            "Tiktoken module not installed. Please add tiktoken to requirements.txt and restart the server."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
