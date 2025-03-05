from unittest.mock import patch

import pytest
from gensim.models import KeyedVectors

# Constants for test cases
HTTP_200 = 200
HTTP_404 = 404
HTTP_500 = 500
EMBEDDING_SIZE = 300
DEFAULT_SIMILAR_COUNT = 10
CUSTOM_SIMILAR_COUNT = 5


def test_root(client):
    response = client.get("/")
    assert response.status_code == HTTP_200
    assert response.json() == {
        "message": "GloVe Word Embeddings API",
        "model": "glove-wiki-gigaword-300",
        "dimensions": 300,
        "status": "running"
    }


def test_get_embedding(client):
    # Test successful case
    response = client.post("/embedding", json={"word": "hello"})
    assert response.status_code == HTTP_200
    data = response.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) == EMBEDDING_SIZE


def test_tokenize(client):
    """Test the tokenize endpoint with a simple string."""
    response = client.post("/tokenize", json={"text": "Hello, world!", "model": "gpt-3.5-turbo"})
    assert response.status_code == HTTP_200
    data = response.json()
    assert "tokens" in data
    assert "token_count" in data
    assert "token_strings" in data
    assert data["token_count"] > 0
    assert len(data["tokens"]) == data["token_count"]
    assert len(data["token_strings"]) == data["token_count"]


def test_available_tokenizers(client):
    """Test the available-tokenizers endpoint."""
    response = client.get("/available-tokenizers")
    assert response.status_code == HTTP_200
    data = response.json()
    assert "available_models" in data
    assert "default_encoding" in data
    assert len(data["available_models"]) > 0
    assert "gpt-3.5-turbo" in data["available_models"]
    assert "gpt-4" in data["available_models"]


    # Test word not in vocabulary
    response = client.post("/embedding", json={"word": "thisisnotarealword"})
    assert response.status_code == HTTP_404

    # Test internal server error
    with patch.object(KeyedVectors, "__getitem__", side_effect=Exception("Test error")):
        response = client.post("/embedding", json={"word": "hello"})
        assert response.status_code == HTTP_500
        assert "Test error" in response.json()["detail"]


def test_get_embeddings(client):
    # Test successful case
    response = client.post("/embeddings", json={"words": ["hello", "world"]})
    assert response.status_code == HTTP_200
    data = response.json()
    assert "results" in data
    assert "hello" in data["results"]
    assert "world" in data["results"]
    assert isinstance(data["results"]["hello"]["embedding"], list)
    assert len(data["results"]["hello"]["embedding"]) == EMBEDDING_SIZE

    # Test with non-existent word
    response = client.post("/embeddings", json={"words": ["hello", "thisisnotarealword"]})
    assert response.status_code == HTTP_200
    data = response.json()
    assert data["results"]["hello"]["embedding"] is not None
    assert data["results"]["thisisnotarealword"]["embedding"] is None

    # Test internal server error
    with patch.object(KeyedVectors, "__getitem__", side_effect=Exception("Test error")):
        response = client.post("/embeddings", json={"words": ["hello"]})
        assert response.status_code == HTTP_200
        assert data["results"]["hello"]["embedding"] is not None


def test_get_similar_words(client):
    # Test successful case
    response = client.get("/similar/hello")
    assert response.status_code == HTTP_200
    data = response.json()
    assert "similar_words" in data
    assert len(data["similar_words"]) == DEFAULT_SIMILAR_COUNT
    assert all(isinstance(item["similarity"], float) for item in data["similar_words"])
    assert all(isinstance(item["word"], str) for item in data["similar_words"])

    # Test with custom n
    response = client.get("/similar/hello?n=5")
    assert response.status_code == HTTP_200
    data = response.json()
    assert len(data["similar_words"]) == CUSTOM_SIMILAR_COUNT

    # Test word not in vocabulary
    response = client.get("/similar/thisisnotarealword")
    assert response.status_code == HTTP_404

    # Test internal server error
    with patch.object(KeyedVectors, "most_similar", side_effect=Exception("Test error")):
        response = client.get("/similar/hello")
        assert response.status_code == HTTP_500
        assert "Test error" in response.json()["detail"]


def test_model_loading_error():
    with patch.object(KeyedVectors, "load", side_effect=Exception("Model loading error")):
        with pytest.raises(Exception, match="Model loading error"):
            # This will trigger the model loading code in app.py
            from importlib import reload

            import app

            reload(app)
