import pytest
import numpy as np

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "GloVe Word Embeddings API"
    assert data["model"] == "glove-wiki-gigaword-300"
    assert data["dimensions"] == 300
    assert data["status"] == "running"

def test_get_embedding(client):
    # Test successful case
    response = client.post("/embedding", json={"word": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) == 300  # GloVe dimension

    # Test word not in vocabulary
    response = client.post("/embedding", json={"word": "thisisnotarealword"})
    assert response.status_code == 404

def test_get_embeddings(client):
    # Test successful case
    response = client.post("/embeddings", json={"words": ["hello", "world"]})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "hello" in data["results"]
    assert "world" in data["results"]
    assert isinstance(data["results"]["hello"]["embedding"], list)
    assert len(data["results"]["hello"]["embedding"]) == 300

    # Test with non-existent word
    response = client.post("/embeddings", json={"words": ["hello", "thisisnotarealword"]})
    assert response.status_code == 200
    data = response.json()
    assert data["results"]["hello"]["embedding"] is not None
    assert data["results"]["thisisnotarealword"]["embedding"] is None

def test_get_similar_words(client):
    # Test successful case
    response = client.get("/similar/hello")
    assert response.status_code == 200
    data = response.json()
    assert "similar_words" in data
    assert len(data["similar_words"]) == 10  # default n=10
    assert all(isinstance(item["similarity"], float) for item in data["similar_words"])

    # Test with custom n
    response = client.get("/similar/hello?n=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data["similar_words"]) == 5

    # Test word not in vocabulary
    response = client.get("/similar/thisisnotarealword")
    assert response.status_code == 404 