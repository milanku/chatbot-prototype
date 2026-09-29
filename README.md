# Chatbot Prototype

A command-line banking assistant that combines transaction Q&A with retrieval-augmented generation (RAG) over local banking documentation.

An OpenAI model routes each message to the appropriate workflow. Depending on the request, the chatbot either works with mocked transaction data or answers from the local document corpus using retrieval, reranking, evidence selection, answer synthesis, and claim verification.

## What it does

- Calculates transaction summaries from natural-language questions and timeframes.
- Lists matching transactions from a local JSON fixture.
- Keeps transaction-summary state in memory during the current CLI session so follow-up questions can refer to previous results.
- Answers banking-document questions using:
  - embedding-based semantic retrieval;
  - BM25 keyword retrieval;
  - reranking;
  - LLM-based required-evidence selection;
  - answer synthesis;
  - claim extraction and verification.
- Writes structured logs with session and trace IDs under `logs/`.
- Includes separate RAG and prompt evaluation tooling under `rag_eval/` and `prompt_evals/`.

The current user interface is an interactive CLI. Transaction data is mocked and session state is stored in memory.

## Requirements

Before installing the project, make sure you have:

- Git
- Python 3.11, 3.12, or 3.13
- [Poetry](https://python-poetry.org/docs/#installation)
- An OpenAI API key
- Internet access during installation
- Internet access on the first document-Q&A request when using the default local embedding and reranking models, because Hugging Face model files may need to be downloaded

Check the installed versions:

```bash
git --version
python3.11 --version
poetry --version
```

The project currently declares:

```text
Python >=3.11,<3.14
```

If `python3.11` is not available, use another supported interpreter such as `python3.12` or `python3.13`.

## Quick start

For a fresh machine, the shortest setup is:

```bash
git clone https://github.com/milanku/chatbot-prototype.git
cd chatbot-prototype

poetry config virtualenvs.in-project true --local
poetry env use python3.11
poetry install

cp .env.example .env
# Edit .env and set OPENAI_API_KEY

source .venv/bin/activate
python -m bot --verbose
```

## Installation

Clone the repository:

```bash
git clone https://github.com/milanku/chatbot-prototype.git
cd chatbot-prototype
```

Configure Poetry to create the virtual environment inside the repository, then select the Python interpreter:

```bash
poetry config virtualenvs.in-project true --local
poetry env use python3.11
```

If your Poetry installation is already configured globally with `virtualenvs.in-project = true`, the first command is optional.

Install the project and its dependencies:

```bash
poetry install
```

The project is configured to use an in-project virtual environment named `.venv/`.

Activate it:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

You can verify the interpreter with:

```bash
python --version
```

and inspect the Poetry environment with:

```bash
poetry env info
```

## Environment configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env` and set the required values.

Current template:

```dotenv
# LLM provider
OPENAI_API_KEY=insert_your_openai_key_here
OPENAI_LLM_PROVIDER=openai
DEFAULT_OPENAI_LLM_MODEL=gpt-4.1-mini

# Paths
DOCS_DIR_PATH=data/docs
TRANSACTIONS_MOCK_FILE_PATH=data/mocks/transactions_mock_jan2025_sep2026.json
```

At minimum, replace:

```dotenv
OPENAI_API_KEY=insert_your_openai_key_here
```

with a valid OpenAI API key.

Do not commit the `.env` file. It is ignored by Git.

### Environment-backed paths

Runtime paths are loaded through:

```text
src/bot/app_settings.py
```

The currently supported path settings are:

- `DOCS_DIR_PATH` — directory containing Markdown documents for document Q&A.
- `TRANSACTIONS_MOCK_FILE_PATH` — JSON fixture used by transaction Q&A.

Both have defaults in `AppSettings`, so the repository works with the existing `data/` layout unless you want to point the application somewhere else.

For example:

```dotenv
DOCS_DIR_PATH=/absolute/path/to/my/docs
TRANSACTIONS_MOCK_FILE_PATH=/absolute/path/to/transactions.json
```

## Runtime data

With the default configuration, the application expects:

```text
data/
├── docs/
│   └── *.md
├── embeddings/
└── mocks/
    └── transactions_mock_jan2025_sep2026.json
```

### Documents

Markdown files under:

```text
data/docs/
```

are used as the RAG knowledge base.

To test different documentation, either replace/add Markdown files in that directory or point `DOCS_DIR_PATH` to another directory in `.env`.

### Transactions

Transaction questions use:

```text
data/mocks/transactions_mock_jan2025_sep2026.json
```

by default.

To use another fixture, update:

```dotenv
TRANSACTIONS_MOCK_FILE_PATH=...
```

### Embeddings

Generated embeddings are persisted under:

```text
data/embeddings/
```

This directory is ignored by Git.

The document pipeline loads an existing embeddings store when possible and otherwise creates one from the configured documents, chunker, and embedding model.

## Run the chatbot

Activate the virtual environment if it is not already active:

```bash
source .venv/bin/activate
```

Run the chatbot from the repository root:

```bash
python -m bot
```

For detailed structured logging:

```bash
python -m bot --verbose
```

or:

```bash
python -m bot -v
```

If you prefer not to activate the virtual environment, use:

```bash
poetry run python -m bot --verbose
```

Example questions:

```text
How much did I spend on pets last month?
Show me my transactions from February 2026.
Which transactions made up that total?
What documents do I need to open an account?
How long does a SEPA transfer take?
```

Type:

```text
exit
```

or:

```text
quit
```

to stop the session.

## Document Q&A pipeline

The document-Q&A feature is initialized lazily. Local embedding and reranking models are therefore loaded only when a request is routed to documentation Q&A.

The current pipeline is:

```text
Question
  ↓
Embedding retrieval + BM25 retrieval
  ↓
Merge / deduplicate candidates
  ↓
Reranker
  ↓
LLM required-evidence judge
  ↓
Answer synthesis
  ↓
Claim extraction
  ↓
Claim verification
  ↓
Answer
```

The main composition logic lives in:

```text
src/bot/composition/doc_qa.py
```

## Runtime configuration

Most application-level parameters are defined in:

```text
src/bot/config/bot.py
```

The current configuration looks conceptually like:

```python
BOT_CONFIG = BotConfig(
    docs_dir_path=...,
    embeddings_dir_path=Path("data/embeddings"),
    transactions_mock_file_path=...,
    llm=OpenAILLMConfig(
        model="gpt-4.1-mini",
    ),
    embedder=LocalEmbedderConfig(
        model=SupportedLocalEmbedder.JINA_EMBEDDINGS_V3,
    ),
    chunker=ContextualChunkerConfig(
        version="contextual_chunker_v01",
    ),
    retrievers=[
        EmbeddingsRetrieverConfig(top_k=15),
        BM25RetrieverConfig(top_k=10),
    ],
    reranker=LocalRerankerConfig(
        model=SupportedLocalReranker.BGE_RERANKER_V2_M3,
        use_fp16=True,
        top_k=7,
    ),
)
```

### LLM model

The default chat model is configured in:

```text
src/bot/config/bot.py
```

Current value:

```python
OpenAILLMConfig(
    model="gpt-4.1-mini",
)
```

Change the model there if you want to test a different OpenAI chat model supported by the current client implementation.

### Embedding model

Embedding configuration is defined in:

```text
src/bot/config/embedder.py
```

Currently supported local models include:

```text
jinaai/jina-embeddings-v3
qwen/Qwen3-Embedding-4B
```

The default application configuration uses:

```python
LocalEmbedderConfig(
    model=SupportedLocalEmbedder.JINA_EMBEDDINGS_V3,
)
```

OpenAI embedding configuration is also supported by the config layer:

```text
text-embedding-3-small
text-embedding-3-large
```

To switch embedder implementation, update the `embedder` field in `src/bot/config/bot.py` and import the corresponding config/model enum.

Changing the embedding model may cause the embeddings store to be rebuilt.

### Retrieval parameters

Retriever settings are defined in:

```text
src/bot/config/retriever.py
```

The default application configuration is:

```python
retrievers=[
    EmbeddingsRetrieverConfig(top_k=15),
    BM25RetrieverConfig(top_k=10),
]
```

These values control how many candidates each retriever contributes before reranking.

For example:

```python
retrievers=[
    EmbeddingsRetrieverConfig(top_k=20),
    BM25RetrieverConfig(top_k=15),
]
```

will increase the candidate pool.

Higher values may improve recall but also increase reranking cost and latency.

### Reranker

The reranker configuration is defined in:

```text
src/bot/config/reranker.py
```

The currently supported reranker is:

```text
BAAI/bge-reranker-v2-m3
```

Default configuration:

```python
reranker=LocalRerankerConfig(
    model=SupportedLocalReranker.BGE_RERANKER_V2_M3,
    use_fp16=True,
    top_k=5,
)
```

Important parameters:

- `top_k` — number of highest-ranked chunks passed to the evidence judge.
- `use_fp16` — enables half-precision execution where supported.

If your machine does not support FP16 correctly, change:

```python
use_fp16=False
```

### Chunking

Chunking configuration is defined in:

```text
src/bot/config/chunker.py
```

The current version is:

```python
ContextualChunkerConfig(
    version="contextual_chunker_v01",
)
```

The version is also part of embeddings-store identity. If you change chunking behavior, update the version so an incompatible existing embeddings store is not silently reused.

### Embeddings directory

The embeddings directory is currently configured directly in:

```text
src/bot/config/bot.py
```

as:

```python
embeddings_dir_path=Path("data/embeddings")
```

Unlike the document and transaction paths, this value is not currently read from `.env`.

## Prompt configuration

Prompt versions are configured in:

```text
src/bot/config/prompts_config.py
```

The application independently versions prompts for:

- route selection;
- required-chunk judging;
- document answer synthesis;
- claim extraction;
- claim verification;
- transaction-summary explanation parsing;
- timeframe parsing.

For example:

```python
claim_extractor=PromptConfig(
    directory=Path("doc_qa/claim_extractor_instructions"),
    version="v002",
)
```

To test a new prompt version:

1. Add the new prompt file in the corresponding prompt directory.
2. Change the `version` in `PROMPT_CONFIGS`.
3. Restart the application.

Keeping prompt files versioned makes it easier to compare behavior and evaluation results without overwriting previous prompt versions.

## Logging

Logs are written under:

```text
logs/
```

The logger keeps a complete per-trace log and also organizes logs by severity.

Typical structure:

```text
logs/
├── all/
│   └── <session_id>/
│       └── <trace_id>.log
├── debug/
├── info/
├── warning/
└── error/
```

Each record contains fields such as:

```text
timestamp
session_id
trace_id
log_level
event
payload
```

Embeddings are intentionally omitted from structured log payloads.

Use:

```bash
python -m bot --verbose
```

to print detailed structured logs to the terminal while the chatbot runs.

## Development

Activate the environment first:

```bash
source .venv/bin/activate
```

### Tests

Run the test suite:

```bash
pytest
```

or without activating the environment:

```bash
poetry run pytest
```

Pytest is configured to collect tests from:

```text
tests/
```

### Linting and type checking

Run both Ruff and mypy:

```bash
make check
```

Run Ruff only:

```bash
make lint
```

Run mypy only:

```bash
make typecheck
```

The `makefile` expects the project-local virtual environment at:

```text
.venv/
```

### Pre-commit hooks

Install the configured Git hooks:

```bash
pre-commit install
```

Run all hooks manually:

```bash
pre-commit run --all-files
```

## Evaluation tooling

The repository also contains experimental and offline evaluation code.

### RAG evaluation

```text
rag_eval/
```

contains tooling and artifacts for evaluating retrieval, reranking, and evidence-selection behavior.

### Prompt evaluation

```text
prompt_evals/
```

contains prompt-specific evaluation utilities, including claim-verification evaluation.

These are separate from the main CLI runtime.

## Project structure

A simplified overview:

```text
chatbot-prototype/
├── data/
│   ├── docs/
│   ├── embeddings/
│   └── mocks/
├── prompt_evals/
├── rag_eval/
├── src/
│   └── bot/
│       ├── composition/
│       ├── config/
│       ├── doc_qa/
│       ├── handlers/
│       ├── interfaces/
│       ├── llm/
│       └── tx_qa/
├── tests/
├── .env.example
├── makefile
├── poetry.lock
└── pyproject.toml
```
