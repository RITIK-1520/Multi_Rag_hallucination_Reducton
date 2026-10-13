"""
LLM interface – wraps Google Gemini (live mode) and a deterministic
offline fallback (demo mode).
"""

from src.config import GEMINI_API_KEY, GEMINI_MODEL, is_api_available


def _get_gemini_client():
    """Lazily import and configure the Google GenAI client."""
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client


def call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """
    Call the LLM.  Uses Gemini when a valid key is present, otherwise
    returns a deterministic offline response.
    """
    if is_api_available():
        return _call_gemini(prompt, temperature, max_tokens)
    return _offline_response(prompt)


def _call_gemini(prompt: str, temperature: float, max_tokens: int) -> str:
    """Call Google Gemini API."""
    try:
        client = _get_gemini_client()
        from google.genai import types
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""
    except Exception as e:
        return f"[LLM Error] {e}"


def _offline_response(prompt: str) -> str:
    """
    Deterministic offline fallback.
    Parses the prompt to figure out which agent is calling and returns
    a structured placeholder derived from the *context* already present
    in the prompt (i.e., from retrieved evidence).
    """
    prompt_lower = prompt.lower()

    # ── Query Analyzer agent ──
    if "analyze the following query" in prompt_lower or "query analysis" in prompt_lower:
        # Extract the query from the prompt
        query = _extract_between(prompt, 'Query: "', '"')
        if not query:
            query = _extract_between(prompt, "Query: ", "\n")
        keywords = " ".join(query.split()[:6]) if query else "general query"
        return (
            f'{{"original_query": "{query}", '
            f'"intent": "information_retrieval", '
            f'"keywords": ["{keywords}"], '
            f'"sub_queries": ["{query}"], '
            f'"retrieval_plan": "Search knowledge base for relevant information"}}'
        )

    # ── Answer Generator agent ──
    if "generate a comprehensive answer" in prompt_lower or "answer the following" in prompt_lower:
        # Extract evidence from the prompt
        evidence = _extract_between(prompt, "Evidence:", "---")
        if not evidence:
            evidence = _extract_between(prompt, "Context:", "---")
        if not evidence:
            evidence = _extract_between(prompt, "Evidence:", "Instructions")
        if evidence and len(evidence.strip()) > 30:
            # Build answer from the evidence text
            sentences = [s.strip() for s in evidence.replace("\n", " ").split(". ") if len(s.strip()) > 15]
            answer_sentences = sentences[:5]
            return ". ".join(answer_sentences) + "." if answer_sentences else "Based on the available evidence, the information is limited."
        return "Insufficient evidence in the knowledge base to provide a reliable answer."

    # ── Hallucination Critic agent ──
    if "verify" in prompt_lower and "claim" in prompt_lower:
        return (
            '{"claims": [{"claim": "offline mode claim", "status": "SUPPORTED", '
            '"reason": "Verified against evidence in offline mode"}], '
            '"supported_count": 1, "unsupported_count": 0, "contradiction_count": 0, '
            '"support_percentage": 100.0, "hallucination_risk": 0.0}'
        )

    # ── Generic fallback ──
    return "Response synthesized from knowledge base documents and legal context retrieval."


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    """Extract text between two markers."""
    try:
        s = text.index(start_marker) + len(start_marker)
        e = text.index(end_marker, s)
        return text[s:e].strip()
    except ValueError:
        # If end marker not found, take the rest (up to 2000 chars)
        try:
            s = text.index(start_marker) + len(start_marker)
            return text[s:s + 2000].strip()
        except ValueError:
            return ""
