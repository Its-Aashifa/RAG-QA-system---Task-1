"""
LLM answer generation using Groq API (LLaMA 3 70B).

Why Groq?
  - You already use Groq API in your existing projects (familiarity)
  - Ultra-low latency inference (~200ms for short answers)
  - Free tier is sufficient for prototyping
  - LLaMA 3 70B has strong instruction-following for RAG prompts

Improvements over basic RAG:
  1. Query rewriting — before retrieval, the LLM rephrases the user's question
     into a more retrieval-friendly form. This helps when users ask vague or
     conversational questions that don't match document vocabulary well.
  2. Confidence scoring — the LLM self-reports a confidence level (high/medium/low)
     based on how well the retrieved context actually answers the question.
     This helps downstream systems decide whether to surface a fallback.
  3. Grounded system prompt — model is strictly forbidden from using prior knowledge.
"""

import logging
import time
from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Query rewriting prompt ────────────────────────────────────────────────────
REWRITE_SYSTEM_PROMPT = """You are a search query optimizer for a document retrieval system.
Rewrite the user's question into a concise, keyword-rich search query that will 
retrieve the most relevant document chunks. 
- Remove conversational filler ("can you tell me", "I want to know")
- Keep domain-specific terms exactly as written
- Return ONLY the rewritten query, nothing else."""

# ── Answer generation prompt ──────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a precise document question-answering assistant.
Answer the user's question using ONLY the provided context chunks.

Rules:
1. If the context fully answers the question, answer clearly and cite the source number e.g. [1].
2. If the context partially answers, answer what you can and note what is missing.
3. If the context does not contain enough information, say exactly:
   "I could not find a sufficient answer in the uploaded documents."
4. Do NOT use prior knowledge outside the context.
5. End your response with a confidence assessment on a new line:
   CONFIDENCE: high | medium | low
   - high = context directly and completely answers the question
   - medium = context partially answers or requires inference
   - low = context is barely relevant or question cannot be answered"""

RAG_USER_TEMPLATE = """Context chunks (from most to least relevant):
{context}

Original question: {original_question}
Optimized search query used: {rewritten_query}

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


def rewrite_query(question: str, client) -> str:
    """
    Rewrite the user's question into a retrieval-optimised search query.

    Example:
      Input:  "Can you tell me what happens when seeds don't get water?"
      Output: "seed germination water requirement drought effect"

    Falls back to original question on any error.
    """
    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info(f"Query rewrite: '{question}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")
        return question


def parse_confidence(answer_text: str) -> tuple[str, str]:
    """
    Extract the CONFIDENCE line from the LLM response.
    Returns (clean_answer, confidence_level).
    """
    lines = answer_text.strip().split("\n")
    confidence = "medium"  # default
    clean_lines = []

    for line in lines:
        if line.strip().upper().startswith("CONFIDENCE:"):
            raw = line.split(":", 1)[1].strip().lower()
            if "high" in raw:
                confidence = "high"
            elif "low" in raw:
                confidence = "low"
            else:
                confidence = "medium"
        else:
            clean_lines.append(line)

    return "\n".join(clean_lines).strip(), confidence


def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate an answer using the LLM given a question and retrieved chunks.

    Pipeline:
      1. Rewrite query for better retrieval alignment (logged, not re-retrieved here
         but returned so callers can log/display it)
      2. Build context from chunks
      3. Generate grounded answer with confidence self-assessment
      4. Parse and return structured result

    Returns:
        dict with keys: answer, rewritten_query, confidence, latency_ms, model
    """
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — returning raw context as answer.")
        context = build_context_string(retrieved_chunks)
        return {
            "answer": f"[LLM not configured] Relevant excerpts:\n\n{context}",
            "rewritten_query": question,
            "confidence": "unknown",
            "latency_ms": 0.0,
            "model": "none",
        }

    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq not installed. Run: pip install groq")

    client = Groq(api_key=settings.GROQ_API_KEY)

    t_start = time.time()

    # Step 1: Rewrite query
    rewritten = rewrite_query(question, client)

    # Step 2: Build context
    context = build_context_string(retrieved_chunks)
    user_message = RAG_USER_TEMPLATE.format(
        context=context,
        original_question=question,
        rewritten_query=rewritten,
    )

    # Step 3: Generate answer
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    latency_ms = (time.time() - t_start) * 1000

    raw_answer = response.choices[0].message.content.strip()

    # Step 4: Parse confidence
    clean_answer, confidence = parse_confidence(raw_answer)

    logger.info(
        f"LLM answer | model={settings.LLM_MODEL} | "
        f"confidence={confidence} | latency={latency_ms:.0f}ms | "
        f"tokens={response.usage.total_tokens}"
    )

    return {
        "answer": clean_answer,
        "rewritten_query": rewritten,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "model": settings.LLM_MODEL,
    }
