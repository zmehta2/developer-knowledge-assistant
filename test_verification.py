#!/usr/bin/env python3
"""
test_springboot_angular.py

End-to-end test of the codebase QA system against
https://github.com/zmehta2/Spring-Boot-Angular-8-CRUD-Example.git

This script does the following:

1. Clones the Git repository (if already present, it pulls the latest changes).
2. Scans the repository for source files (.java, .ts, .js, .html, .css).
3. Parses the files into logical code units (classes, methods, or generic text chunks).
4. Generates embeddings for a subset of the chunks using a CodeBERT model.
5. Stores these embeddings in a local vector store.
6. Runs a few example natural-language queries and prints the top matches.
7. Prints a short process summary at the end, so the behavior is easy to explain.
"""

import sys
from pathlib import Path

from src.embeddings.code_embedder import CodeEmbedder

# Make src importable
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main() -> None:
    print("Codebase QA - Spring Boot + Angular Example")
    print("-" * 60)

    # ------------------------------------------------------------------
    # Step 1: Clone repository and list files
    # ------------------------------------------------------------------
    from ingestion.git_crawler import GitCrawler

    repo_url = "https://github.com/zmehta2/Spring-Boot-Angular-8-CRUD-Example.git"
    crawler = GitCrawler(data_dir="./data/raw")

    print("\nStep 1: Cloning repository")
    print("-" * 60)
    repo_path = crawler.clone_repo(repo_url)
    print(f"Repository: {repo_url}")
    print(f"Local path: {repo_path}")

    extensions = [".java", ".ts", ".js", ".html", ".css"]
    files = crawler.get_code_files(repo_path, extensions=extensions)

    print(f"\nSource files found: {len(files)}")
    ext_counts = {}
    for f in files:
        ext = f["extension"]
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"  {ext}: {count} files")

    # ------------------------------------------------------------------
    # Step 2: Parse files into chunks
    # ------------------------------------------------------------------
    from ingestion.code_parser import MultiLanguageParser

    print("\nStep 2: Parsing code files into chunks")
    print("-" * 60)

    parser = MultiLanguageParser()
    all_chunks = []

    for f in files:
        path = f["path"]
        content = f["content"]
        try:
            parsed = parser.parse_file(content, path)
            chunks = parser.create_searchable_chunks(parsed, content)
            
            for c in chunks:
                # Make sure metadata exists
                if "metadata" not in c or c["metadata"] is None:
                    c["metadata"] = {}
                c["metadata"]["path"] = path
                c["metadata"]["extension"] = f["extension"]
            
            all_chunks.extend(chunks)
            if chunks:
                print(f"  {path}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  {path}: parser error ({e})")

    print(f"\nTotal chunks created: {len(all_chunks)}")

    type_counts = {}
    for c in all_chunks:
        t = c.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print("Chunk types:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # Show a few representative chunks for Java and TypeScript
    java_chunks = [
        c for c in all_chunks
        if c.get("metadata", {}).get("language") == "java"
    ]
    ts_chunks = [
        c for c in all_chunks
        if c.get("metadata", {}).get("language") == "typescript"
    ]

    print("\nExample Java chunks:")
    for c in java_chunks[:5]:
        print(f"  {c.get('type')}: {c.get('name')}")

    print("\nExample TypeScript chunks:")
    for c in ts_chunks[:5]:
        print(f"  {c.get('type')}: {c.get('name')}")

    # ------------------------------------------------------------------
    # Step 3: Embeddings
    # ------------------------------------------------------------------
    print("\nStep 3: Generating embeddings for chunks")
    print("-" * 60)

    if not all_chunks:
        print("No chunks available; stopping here.")
        return

    from embeddings.code_embedder import CodeEmbedder

    embedder = CodeEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # For speed, only embed the first 50 chunks
    chunks_to_embed = [c for c in all_chunks if c.get('type') != 'text_chunk']

    print(f"Chunks selected for embedding: {len(chunks_to_embed)}")

    chunks_with_emb = embedder.embed_code_chunks(chunks_to_embed)

    # Assume each embedding is a 1D vector; get dimension from first one
    first_emb = chunks_with_emb[0].get("embedding")
    emb_dim = getattr(first_emb, "shape", None)
    if emb_dim is not None and len(emb_dim) > 0:
        embedding_dim = emb_dim[-1]
    else:
        embedding_dim = "unknown"

    print(f"Embedding dimension: {embedding_dim}")

    # ------------------------------------------------------------------
    # Step 4: Store in vector database
    # ------------------------------------------------------------------
    print("\nStep 4: Storing embeddings in vector store")
    print("-" * 60)

    from search.vector_store import CodeVectorStore

    store = CodeVectorStore(persist_dir="./data/springboot_chroma")
    store.clear()
    store.add_chunks(chunks_with_emb)

    # Some implementations expose a collection.count() method; guard it
    try:
        vs_count = store.collection.count()
    except Exception:
        vs_count = len(chunks_with_emb)

    print(f"Items stored in vector store: {vs_count}")

    # ------------------------------------------------------------------
    # Step 5: Run example queries
    # ------------------------------------------------------------------
    print("\nStep 5: Testing semantic search with example queries")
    print("-" * 60)

    test_queries = [
        "EmployeeController class REST API",
        "EmployeeListComponent Angular",
        "EmployeeRepository JpaRepository interface",
        "CreateEmployeeComponent form",
    ]

    example_matches = []

    for query in test_queries:
        query_emb = embedder.embed_text(query)
        # results = store.search(query, query_emb, n_results=3)
        
        results = store.search(query, query_emb, n_results=3, filter_dict={'type': 'method'})

        print(f"\nQuery: {query}")
        if not results:
            print("  No results returned.")
            continue

        for i, r in enumerate(results, start=1):
            md = r.get("metadata", {})
            name = md.get("name", "")
            ctype = md.get("type", "")
            path = md.get("file", "(no file)")
            score = r.get("score", None)
            if score is not None:
                print(f"  {i}. {name} ({ctype}) - score {score:.3f}")
            else:
                print(f"  {i}. {name} ({ctype})")

        # Keep only the top result for the final summary
        top = results[0]
        top_md = top.get("metadata", {})
        example_matches.append(
            {
                "query": query,
                "name": top_md.get("name", ""),
                "type": top_md.get("type", ""),
                "path": top_md.get("file", ""),  
            }
        )

    # ------------------------------------------------------------------
    # Step 6: Process overview (plain explanation)
    # ------------------------------------------------------------------
    print("\nSummary of this run")
    print("-" * 60)

    print(f"Repository URL: {repo_url}")
    print(f"Local path:     {repo_path}")
    print(f"Files scanned:  {len(files)}")
    print(f"Total chunks:   {len(all_chunks)}")
    print(f"Embedded:       {len(chunks_to_embed)}")
    print(f"Stored in DB:   {vs_count}")

    print("\nProcess overview:")
    print("1. The script cloned the Git repository and collected source files")
    print("   with extensions .java, .ts, .js, .html, and .css.")
    print("2. Each file was parsed into smaller units such as classes, methods,")
    print("   or generic text chunks, which are easier to search individually.")
    print("3. For a subset of chunks, the script computed numerical embeddings")
    print("   using a CodeBERT model. These vectors represent the meaning of")
    print("   the code units.")
    print("4. The embeddings and metadata were stored in a local vector store,")
    print("   which can efficiently return the closest chunks for a given query.")
    print("5. For each example natural-language query above, the script embedded")
    print("   the query, searched in the vector store, and printed the closest")
    print("   matching code elements.")

    print("\nThis script only builds and tests the search index.")
    print("A separate component can use this index to construct full answers")
    print("for users by combining relevant chunks and explaining them.")


if __name__ == "__main__":
    main()