"""
LLM answer generation using Groq API (LLaMA 3 70B).

Why Groq?
  - You already use Groq API in your existing projects (familiarity)
  - Ultra-low latency inference (~200ms for short answers)
  - Free tier is sufficient for prototyping
  - LLaMA 3 70B has strong instruction-following for RAG prompts

The prompt is designed to:
  1. Ground the model strictly in retrieved context (reduce hallucination)
  2. Ask the model to cite which source it used
  3. Gracefully handle cases where context is insufficient
"""

import logging
import time
from app.core.config import settings

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a precise document question-answering assistant.
Answer the user's question using ONLY the provided context chunks.
If the context does not contain enough information to answer, say:
"I could not find a sufficient answer in the uploaded documents."
Do not use prior knowledge. Be concise and accurate."""

RAG_USER_TEMPLATE = """Context chunks (from most to least relevant):
{context}

Question: {question}

Answer:"""


def build_context_string(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk['source']} (chunk {chunk['chunk_index']}, "
            f"score: {chunk['similarity_score']:.3f})\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate an answer using the LLM given a question and retrieved chunks.

    Returns:
        dict with keys: answer (str), latency_ms (float), model (str)
    """
    if not settings.GROQ_API_KEY:
        # Fallback: return context directly if no LLM key configured
        logger.warning("GROQ_API_KEY not set — returning raw context as answer.")
        context = build_context_string(retrieved_chunks)
        return {
            "answer": f"[LLM not configured] Relevant excerpts:\n\n{context}",
            "latency_ms": 0.0,
            "model": "none",
        }

    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq not installed. Run: pip install groq")

    context = build_context_string(retrieved_chunks)
    user_message = RAG_USER_TEMPLATE.format(context=context, question=question)

    client = Groq(api_key=settings.GROQ_API_KEY)

    t_start = time.time()
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,        # low temp for factual RAG
        max_tokens=1024,
    )
    latency_ms = (time.time() - t_start) * 1000

    answer = response.choices[0].message.content.strip()

    logger.info(
        f"LLM answer generated | model={settings.LLM_MODEL} | "
        f"latency={latency_ms:.0f}ms | tokens={response.usage.total_tokens}"
    )

    return {
        "answer": answer,
        "latency_ms": latency_ms,
        "model": settings.LLM_MODEL,
    }
