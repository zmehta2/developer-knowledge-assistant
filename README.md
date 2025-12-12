# developer-knowledge-assistant
A Transformer-based semantic search and question-answering tool for large codebases.

# test_verification

This file documents the `test_verification.py` script included in this repository. The script performs a small end-to-end run of the codebase QA pipeline against a sample repository (Spring Boot + Angular). It clones the repository, parses source files, creates embeddings, stores them in a local vector store, and runs example semantic queries.

## Files

- `test_verification.py` — the runnable script that drives the demo pipeline.

## Purpose

- Demonstrate the ingestion -> parsing -> embedding -> vector-store -> search flow.
- Provide a reproducible smoke test that can be used during development.

## Prerequisites

- Python 3.9 or newer (3.11 recommended)
- Git installed and on your PATH
- Network access to clone repositories and download model weights
- Install the repository Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quickstart

1. From the repository root, run:

```bash
python test_verification.py
```

2. The script will:
   - Clone the example repository into `./data/raw` (or pull if already cloned).
   - Scan for files with extensions: `.java`, `.ts`, `.js`, `.html`, `.css`.
   - Parse files into chunks (classes, methods, or text chunks).
   - Generate embeddings for a subset of chunks and store them in `./data/springboot_chroma`.
   - Run several sample natural-language queries and print the top matches functions.

## Configuration & customization

- To change the target repository, edit the `repo_url` variable in `test_verification.py`.
- To change file extensions scanned, modify the `extensions` list.
- To change the embedding model, modify the `CodeEmbedder(model_name=...)` call in the script.
- To change the vector store location, edit the `CodeVectorStore(persist_dir=...)` call.

## Notes

- The script uses an off-the-shelf sentence-transformers model by default (`sentence-transformers/all-MiniLM-L6-v2`) but you can switch model to (`microsoft/codebert-base`) and (`microsoft/graphcodebert-base`). Model download will occur on first run.
- The script intentionally limits embeddings to a subset of chunks for speed in the demo; extend or batch as needed for larger projec

## Contact

If you need help running or extending this demo, please email me!