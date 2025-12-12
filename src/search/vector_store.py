"""
src/search/vector_store.py

Vector database for semantic code search using ChromaDB
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeVectorStore:
    """Manage vector storage and semantic search"""
    
    def __init__(self, persist_dir: str = "./data/chroma_db"):
        """
        Initialize ChromaDB
        
        Args:
            persist_dir: Directory to persist database
        """
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="code_chunks",
            metadata={
                    "description": "Code chunks with embeddings",  
                    "hnsw:space": "cosine"
                }
        )
        
        logger.info(f"ChromaDB initialized. Current items: {self.collection.count()}")
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Add code chunks to vector store
        
        Args:
            chunks: List of chunks with embeddings
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        # Prepare data for ChromaDB
        ids = [f"{chunk['file']}::{chunk['name']}" for chunk in chunks]
        embeddings = [chunk['embedding'].tolist() for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        
        # Metadata (must be simple types for ChromaDB)
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                'file': chunk['file'],
                'name': chunk['name'],
                'type': chunk['type'],
                'line_start': chunk['line_start'],
                'line_end': chunk['line_end'],
                'code': chunk['code'][:1000]  # Truncate long code
            })
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks. Total: {self.collection.count()}")
    
    def search(self, query: str, query_embedding: np.ndarray, 
               n_results: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Semantic search for code
        
        Args:
            query: Original query text
            query_embedding: Embedding of query
            n_results: Number of results to return
            filter_dict: Optional metadata filter (e.g., {'type': 'function'})
            
        Returns:
            List of matching chunks with scores
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_dict  # Filter by metadata
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'score': 1 - results['distances'][0][i],  # Now correct for cosine
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i]
            })
        
        return formatted_results
    
    def search_by_text(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Text-based search (ChromaDB will embed it)
        
        Args:
            query: Search query
            n_results: Number of results
            
        Returns:
            List of matching chunks
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
       # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'score': 1 - results['distances'][0][i],  # Now correct for cosine
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i]
            })
        
        return formatted_results
    
    def get_by_file(self, file_path: str) -> List[Dict]:
        """Get all chunks from a specific file"""
        results = self.collection.get(
            where={"file": file_path}
        )
        
        formatted_results = []
        for i in range(len(results['ids'])):
            formatted_results.append({
                'id': results['ids'][i],
                'text': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        return formatted_results
    
    def clear(self):
        self.client.delete_collection("code_chunks")
        self.collection = self.client.create_collection(
            "code_chunks",
            metadata={
                "description": "Code chunks with embeddings",
                "hnsw:space": "cosine"
            }
        )
        logger.info("Vector store cleared")


# Example usage
if __name__ == "__main__":
    from code_embedder import CodeEmbedder
    
    # Sample chunks with embeddings
    embedder = CodeEmbedder()
    
    sample_chunks = [
        {
            'file': 'auth/user.py',
            'name': 'authenticate_user',
            'type': 'function',
            'line_start': 10,
            'line_end': 20,
            'text': 'Function: authenticate_user\nAuthenticates user credentials',
            'code': 'def authenticate_user(username, password):\n    return check_password(username, password)'
        },
        {
            'file': 'payment/stripe.py',
            'name': 'process_payment',
            'type': 'function',
            'line_start': 5,
            'line_end': 15,
            'text': 'Function: process_payment\nProcess credit card payment',
            'code': 'def process_payment(amount, card):\n    return stripe.charge(amount, card)'
        }
    ]
    
    # Add embeddings
    chunks_with_emb = embedder.embed_code_chunks(sample_chunks)
    
    # Initialize vector store
    store = CodeVectorStore(persist_dir="./test_chroma")
    
    # Clear existing data
    store.clear()
    
    # Add chunks
    store.add_chunks(chunks_with_emb)
    
    # Test search with embedding
    query = "how to authenticate users"
    query_emb = embedder.embed_text(query)
    
    print(f"\nQuery: '{query}'")
    print("\nTop 2 results:")
    results = store.search(query, query_emb, n_results=2)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['id']} (score: {result['score']:.3f})")
        print(f"   File: {result['metadata']['file']}")
        print(f"   Type: {result['metadata']['type']}")
        print(f"   Text: {result['text'][:100]}...")
    
    # Test filter by type
    print("\n\nSearching only functions:")
    results = store.search(query, query_emb, n_results=5, 
                          filter_dict={'type': 'function'})
    print(f"Found {len(results)} functions")
    
    # Test get by file
    print("\n\nAll chunks from auth/user.py:")
    auth_chunks = store.get_by_file('auth/user.py')
    print(f"Found {len(auth_chunks)} chunks")