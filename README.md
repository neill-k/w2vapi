---
tags:
- glove
- gensim
- fse
---
# GloVe Word Embeddings API

A FastAPI-based REST API for serving GloVe word embeddings based on Wikipedia and Gigaword dataset.

## Features

- Get word embeddings for single words
- Batch process multiple words
- Find similar words based on embedding similarity
- 300-dimensional word vectors
- FastAPI-powered with automatic Swagger documentation

## Prerequisites

1. Install Git LFS:
   ```bash
   # Ubuntu/Debian
   curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   sudo apt-get install git-lfs

   # macOS
   brew install git-lfs

   # Windows
   # Download and install from https://git-lfs.github.com/
   ```

2. Initialize Git LFS:
   ```bash
   git lfs install
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://your-repository-url.git
   cd your-repository-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download model files:
   ```bash
   git lfs pull
   ```

## Running the API

### Local Development
```bash
uvicorn app:app --reload
```

### Production
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

## API Endpoints

- `GET /` - API information
- `POST /embedding` - Get embedding for a single word
- `POST /embeddings` - Get embeddings for multiple words
- `GET /similar/{word}` - Get similar words

## Testing

Run the tests using pytest:
```bash
pytest tests/
```

## Deployment

### Replit Deployment
This project is configured for deployment on Replit. Simply import the repository and click Run.

### Docker Deployment
1. Build the Docker image:
   ```bash
   docker build -t glove-api .
   ```

2. Run the container:
   ```bash
   docker run -p 8080:8080 glove-api
   ```

## Model Information

Pre-trained GloVe vectors based on Wikipedia and Gigaword dataset:
- 6B tokens
- 400K vocab
- 300d vectors
- Uncased

## Working with Large Files

This repository uses Git LFS to handle large files. The following file types are tracked by Git LFS:
- `.model` - Model files
- `.npy` - NumPy array files
- `.bin` - Binary files
- `.vec` - Vector files
- `.gz` - Compressed files

When cloning the repository, make sure to have Git LFS installed and run `git lfs pull` to download the model files.

## References

- [GloVe Project](https://nlp.stanford.edu/projects/glove/)
- [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf)
- [Git LFS Documentation](https://git-lfs.github.com/)

## License

MIT
