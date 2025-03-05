# GloVe Word Embeddings API Documentation

This documentation provides details about the endpoints available in the GloVe Word Embeddings API.

## Base URL

```
https://w-2-vapi-neill-k.replit.app/
```

## Authentication

No authentication is required to use this API.

## Endpoints

### Root Endpoint

Returns basic information about the API.

**URL**: `/`

**Method**: `GET`

**Response**:
```json
{
    "message": "GloVe Word Embeddings API",
    "model": "glove-wiki-gigaword-300",
    "dimensions": 300,
    "status": "running"
}
```

### Health Check

Check if the model is loaded properly and the API is functioning correctly.

**URL**: `/health`

**Method**: `GET`

**Response**:
- If model is loaded:
```json
{
    "status": "healthy",
    "model_loaded": true
}
```
- If model is still initializing:
```json
{
    "status": "initializing",
    "model_loaded": false,
    "message": "Model is still loading"
}
```

### Get Word Embedding

Retrieve the embedding vector for a single word.

**URL**: `/embedding`

**Method**: `POST`

**Request Body**:
```json
{
    "word": "example"
}
```

**Success Response**:
- **Code**: 200 OK
- **Content**:
```json
{
    "embedding": [0.1234, 0.5678, ..., 0.9012]  // 300-dimensional vector
}
```

**Error Responses**:
- **Code**: 404 Not Found
  - **Content**: `{"detail": "Word 'example' not found in vocabulary"}`
- **Code**: 503 Service Unavailable
  - **Content**: `{"detail": "Model is still initializing, please try again later"}`
- **Code**: 500 Internal Server Error
  - **Content**: `{"detail": "Error message"}`

### Get Multiple Word Embeddings

Retrieve embedding vectors for multiple words in a single request.

**URL**: `/embeddings`

**Method**: `POST`

**Request Body**:
```json
{
    "words": ["example", "word", "embeddings"]
}
```

**Success Response**:
- **Code**: 200 OK
- **Content**:
```json
{
    "results": {
        "example": {
            "embedding": [0.1234, 0.5678, ..., 0.9012]  // 300-dimensional vector
        },
        "word": {
            "embedding": [0.2345, 0.6789, ..., 0.0123]  // 300-dimensional vector
        },
        "embeddings": {
            "embedding": [0.3456, 0.7890, ..., 0.1234]  // 300-dimensional vector
        }
    }
}
```

**Note**: If a word is not found in the vocabulary, its embedding will be `null` rather than returning an error:
```json
{
    "results": {
        "example": {
            "embedding": [0.1234, 0.5678, ..., 0.9012]
        },
        "nonexistentword": {
            "embedding": null
        }
    }
}
```

**Error Response**:
- **Code**: 503 Service Unavailable
  - **Content**: `{"detail": "Model is still initializing, please try again later"}`

### Get Similar Words

Find words that are semantically similar to a given word.

**URL**: `/similar/{word}`

**Method**: `GET`

**URL Parameters**:
- `word`: The word to find similar words for

**Query Parameters**:
- `n`: Number of similar words to return (default: 10)

**Example**:
```
/similar/computer?n=5
```

**Success Response**:
- **Code**: 200 OK
- **Content**:
```json
{
    "similar_words": [
        {
            "word": "computers",
            "similarity": 0.8765
        },
        {
            "word": "laptop",
            "similarity": 0.7654
        },
        {
            "word": "machine",
            "similarity": 0.6543
        },
        {
            "word": "pc",
            "similarity": 0.5432
        },
        {
            "word": "software",
            "similarity": 0.4321
        }
    ]
}
```

**Error Responses**:
- **Code**: 404 Not Found
  - **Content**: `{"detail": "Word 'example' not found in vocabulary"}`
- **Code**: 503 Service Unavailable
  - **Content**: `{"detail": "Model is still initializing, please try again later"}`
- **Code**: 500 Internal Server Error
  - **Content**: `{"detail": "Error message"}`

### Tokenization

Tokenize text using OpenAI's tiktoken library (GPT-3/4 tokenizers).

**URL**: `/tokenize`

**Method**: `POST`

**Request Body**:
```json
{
    "text": "Hello, world!",
    "model": "gpt-4"
}
```

**Response**:
```json
{
    "tokens": [15339, 11, 995, 0],
    "token_count": 4,
    "token_strings": ["Hello", ",", " world", "!"]
}
```

### Available Tokenizers

Get a list of available tokenizers from tiktoken.

**URL**: `/available-tokenizers`

**Method**: `GET`

**Response**:
```json
{
    "available_models": [
        "gpt-4",
        "gpt-3.5-turbo",
        "text-davinci-003",
        "text-davinci-002",
        "text-davinci-001",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
        "davinci",
        "curie",
        "babbage",
        "ada"
    ],
    "default_encoding": "cl100k_base"
}
```

## Model Information

This API uses pre-trained GloVe vectors based on Wikipedia and Gigaword dataset:
- 6B tokens
- 400K vocabulary
- 300-dimensional vectors
- Uncased

## Error Handling

The API returns appropriate HTTP status codes along with error messages in the response body:

- **400 Bad Request**: Invalid input parameters
- **404 Not Found**: Word not found in vocabulary
- **500 Internal Server Error**: Server-side error
- **503 Service Unavailable**: Model still initializing

## Usage Examples

### Python

```python
import requests

# Get embedding for a single word
response = requests.post("https://[your-replit-domain]/embedding", 
                        json={"word": "computer"})
embedding = response.json()["embedding"]

# Get embeddings for multiple words
response = requests.post("https://[your-replit-domain]/embeddings", 
                        json={"words": ["computer", "keyboard", "mouse"]})
results = response.json()["results"]

# Get similar words
response = requests.get("https://[your-replit-domain]/similar/computer?n=5")
similar_words = response.json()["similar_words"]
```

### JavaScript

```javascript
// Get embedding for a single word
fetch("https://[your-replit-domain]/embedding", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ word: "computer" })
})
.then(response => response.json())
.then(data => console.log(data.embedding));

// Get embeddings for multiple words
fetch("https://[your-replit-domain]/embeddings", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ words: ["computer", "keyboard", "mouse"] })
})
.then(response => response.json())
.then(data => console.log(data.results));

// Get similar words
fetch("https://[your-replit-domain]/similar/computer?n=5")
.then(response => response.json())
.then(data => console.log(data.similar_words));
```

## Rate Limiting

There are currently no rate limits imposed on this API, but please be considerate with your usage.

## Support

If you encounter any issues or have questions about this API, please create an issue in the GitHub repository or contact the maintainers.