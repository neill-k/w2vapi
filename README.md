---
tags:
- glove
- gensim
- fse
---
# GloVe Word Embeddings API

A FastAPI-based REST API for serving GloVe word embeddings based on Wikipedia and Gigaword dataset.

[![CI/CD Pipeline](https://github.com/{username}/glove-api/actions/workflows/main.yml/badge.svg)](https://github.com/{username}/glove-api/actions/workflows/main.yml)

## Features

- Get word embeddings for single words
- Batch process multiple words
- Find similar words based on embedding similarity
- 300-dimensional word vectors from Hugging Face model hub
- FastAPI-powered with automatic Swagger documentation
- Automated CI/CD pipeline with GitHub Actions
- Docker support
- Code quality checks (Black, isort, Flake8)
- Comprehensive test suite

## Prerequisites

No special prerequisites required. The model will be automatically downloaded from Hugging Face model hub on first startup.

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



## API Documentation

For detailed API documentation, see [API Documentation](docs/api_documentation.md).

### Quick Endpoint Reference

- `GET /` - API information
- `GET /health` - API health check
- `POST /embedding` - Get embedding for a single word
- `POST /embeddings` - Get embeddings for multiple words
- `GET /similar/{word}` - Get similar words


## Development

### Code Quality

Format your code:
```bash
black .
isort .
```

Run linting:
```bash
flake8 .
```

### Testing

Run tests with coverage:
```bash
pytest tests/ -v --cov=app --cov-report=term-missing
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

### CI/CD Pipeline

The project includes a GitHub Actions workflow that:
1. Runs on every push and pull request to main/master branches
2. Checks code formatting with Black
3. Validates imports with isort
4. Runs linting with Flake8
5. Executes the test suite
6. Deploys to Replit (on main/master)
7. Builds and pushes Docker image (on main/master)

Required secrets for deployment:
- `REPL_ID`: Your Replit project ID
- `REPLIT_TOKEN`: Your Replit authentication token
- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token

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
