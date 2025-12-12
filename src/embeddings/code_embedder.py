"""
src/embeddings/code_embedder.py

Generate embeddings for code using CodeBERT or sentence-transformers
"""
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeEmbedder:
    """Generate embeddings for code chunks"""
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """
        Initialize embedder
        
        Args:
            model_name: HuggingFace model name
                Options:
                - "microsoft/codebert-base" (general code)
                - "microsoft/graphcodebert-base" (with data flow)
                - "sentence-transformers/all-MiniLM-L6-v2" (fast, general)
        """
        self.model_name = model_name
        logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text
        
        Args:
            text: Code or text to embed
            
        Returns:
            Embedding vector (numpy array)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use [CLS] token embedding (first token)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of code/text to embed
            batch_size: Process this many at once
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Extract [CLS] embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i+len(batch)}/{len(texts)} texts")
        
        return np.vstack(all_embeddings)
    
    def embed_code_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Add embeddings to code chunks
        
        Args:
            chunks: List of chunk dicts from code_parser
            
        Returns:
            Same chunks with 'embedding' field added
        """
        logger.info(f"Embedding {len(chunks)} code chunks...")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_batch(texts)
        
        # Add to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
            
        logger.info(f"Embeddings complete. Shape: {embeddings.shape}")
        return chunks


# Example usage
if __name__ == "__main__":
    # Sample code chunks
    sample_chunks = [
        {
            'name': 'authenticate_user',
            'text': """Function: authenticate_user
Description: Authenticate user with username and password
Arguments: username, password
Code:
def authenticate_user(username, password):
    hashed = hash_password(password)
    user = db.get_user(username)
    return user and user.password == hashed"""
        },
        {
            'name': 'process_payment',
            'text': """Function: process_payment
Description: Process payment using Stripe
Arguments: amount, card_token
Code:
def process_payment(amount, card_token):
    charge = stripe.Charge.create(
        amount=amount * 100,
        currency='usd',
        source=card_token
    )
    return charge"""
        }
    ]
    
    # Initialize embedder
    embedder = CodeEmbedder(model_name="microsoft/codebert-base")
    
    # Generate embeddings
    chunks_with_embeddings = embedder.embed_code_chunks(sample_chunks)
    
    print(f"\nEmbedding shape: {chunks_with_embeddings[0]['embedding'].shape}")
    print(f"Embedding sample (first 10 dims): {chunks_with_embeddings[0]['embedding'][:10]}")
    
    # Calculate similarity between chunks
    emb1 = chunks_with_embeddings[0]['embedding']
    emb2 = chunks_with_embeddings[1]['embedding']
    
    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"\nSimilarity between authenticate_user and process_payment: {similarity:.3f}")