# Chatbot Prototype

A chatbot prototype with document RAG and transactions Q&A.

## Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

## Setup

1. **Install dependencies**

   ```bash
   poetry install
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set your `OPENAI_API_KEY`.

## Running the chatbot

```bash
PYTHONPATH=src poetry run python -m bot
```

Optional flag:

```bash
PYTHONPATH=src poetry run python -m bot --verbose
```

Type `exit` or `quit` to stop the session.

## Development

**Lint & type-check:**

```bash
make check
```

**Run tests:**

```bash
poetry run pytest
```
