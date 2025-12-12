"""
src/qa/question_answering.py

RAG-based Question Answering over codebase
"""
from typing import List, Dict, Optional
import os
import logging

# Try to import OpenAI, but don't fail if not available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenAI not installed. Install with: pip install openai")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodebaseQA:
    """Question Answering system for code"""
    
    def __init__(self, vector_store, embedder, 
                 use_openai: bool = True,
                 openai_api_key: Optional[str] = None):
        """
        Initialize QA system
        
        Args:
            vector_store: CodeVectorStore instance
            embedder: CodeEmbedder instance
            use_openai: Use OpenAI API (True) or local model (False)
            openai_api_key: OpenAI API key
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.use_openai = use_openai
        
        if use_openai:
            api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("No OpenAI API key provided")
            self.client = OpenAI(api_key=api_key)
    
    def answer_question(self, question: str, 
                       n_context: int = 5,
                       temperature: float = 0.3) -> Dict:
        """
        Answer question about codebase
        
        Args:
            question: User's question
            n_context: Number of code chunks to retrieve
            temperature: LLM temperature (0 = deterministic)
            
        Returns:
            Dict with answer and sources
        """
        # Step 1: Retrieve relevant code chunks
        logger.info(f"Question: {question}")
        
        query_embedding = self.embedder.embed_text(question)
        results = self.vector_store.search(question, query_embedding, n_results=n_context)
        
        if not results:
            return {
                'answer': "I couldn't find any relevant code for that question.",
                'sources': [],
                'confidence': 0.0
            }
        
        logger.info(f"Retrieved {len(results)} relevant chunks")
        
        # Step 2: Build context from retrieved chunks
        context = self._build_context(results)
        
        # Step 3: Generate answer using LLM
        answer = self._generate_answer(question, context, temperature)
        
        # Step 4: Format sources
        sources = self._format_sources(results)
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': results[0]['score'] if results else 0.0
        }
    
    def _build_context(self, results: List[Dict]) -> str:
        """Build context string from search results"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            context_parts.append(
                f"[Source {i}] File: {metadata['file']}\n"
                f"Function/Class: {metadata['name']}\n"
                f"Type: {metadata['type']}\n"
                f"Code:\n{metadata['code']}\n"
            )
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str, 
                        temperature: float) -> str:
        """Generate answer using LLM"""
        
        prompt = f"""You are a helpful assistant that answers questions about a codebase.

Context (relevant code snippets):
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided code snippets
- Explain how the code works in plain English
- Include specific function/class names when relevant
- If the code doesn't fully answer the question, say so
- Keep the answer concise but complete

Answer:"""

        if self.use_openai:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a code expert who explains codebases clearly."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return self._fallback_answer(context)
        else:
            # Fallback: simple extractive answer
            return self._fallback_answer(context)
    
    def _fallback_answer(self, context: str) -> str:
        """Fallback answer when LLM not available"""
        return f"Based on the code, here are the relevant functions:\n\n{context[:500]}..."
    
    def _format_sources(self, results: List[Dict]) -> List[Dict]:
        """Format sources for citation"""
        sources = []
        for result in results:
            metadata = result['metadata']
            sources.append({
                'file': metadata['file'],
                'name': metadata['name'],
                'type': metadata['type'],
                'lines': f"{metadata['line_start']}-{metadata['line_end']}",
                'score': result['score'],
                'code_preview': metadata['code'][:200]
            })
        return sources


class CodeExplainer:
    """Explain what code does"""
    
    def __init__(self, use_openai: bool = True):
        self.use_openai = use_openai
        if use_openai:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def explain_code(self, code: str, context: str = "") -> str:
        """
        Explain what code does in plain English
        
        Args:
            code: Code to explain
            context: Optional context (file name, surrounding code)
            
        Returns:
            Plain English explanation
        """
        prompt = f"""Explain what this code does in simple terms:

{context}

Code:
```
{code}
```

Provide:
1. High-level purpose (1 sentence)
2. Step-by-step explanation
3. Key functions/methods used
4. Potential issues or edge cases

Explanation:"""

        if self.use_openai:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=400
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return "Could not generate explanation (API error)"
        else:
            return "Explanation requires LLM API"


# Example usage
if __name__ == "__main__":
    # Mock setup (you'd use real components)
    from vector_store import CodeVectorStore
    from code_embedder import CodeEmbedder
    
    # Initialize components
    embedder = CodeEmbedder()
    store = CodeVectorStore(persist_dir="./test_chroma")
    
    # Initialize QA system
    qa = CodebaseQA(
        vector_store=store,
        embedder=embedder,
        use_openai=True  # Set to False if no API key
    )
    
    # Ask questions
    questions = [
        "How does authentication work?",
        "How do we process payments?",
        "What libraries are used for database access?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print('='*60)
        
        result = qa.answer_question(question, n_context=3)
        
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nConfidence: {result['confidence']:.2%}")
        print(f"\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['file']} - {source['name']} "
                  f"(lines {source['lines']}, score: {source['score']:.3f})")
    
    # Test code explanation
    print(f"\n{'='*60}")
    print("Code Explanation Example")
    print('='*60)
    
    explainer = CodeExplainer(use_openai=True)
    
    sample_code = """
def process_payment(amount, card_token, customer_id):
    try:
        charge = stripe.Charge.create(
            amount=amount * 100,
            currency='usd',
            source=card_token,
            customer=customer_id,
            metadata={'type': 'subscription'}
        )
        log_transaction(charge.id, customer_id, amount)
        return {'success': True, 'charge_id': charge.id}
    except stripe.error.CardError as e:
        log_error(e)
        return {'success': False, 'error': str(e)}
"""
    
    explanation = explainer.explain_code(sample_code, context="File: payment/stripe.py")
    print(f"\nExplanation:\n{explanation}")