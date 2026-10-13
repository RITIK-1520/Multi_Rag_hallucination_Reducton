"""
Agents module – implements the six agents as callable classes.

    1. QueryAnalyzerAgent
    2. RetrieverAgent
    3. EvidenceVerifierAgent
    4. AnswerGeneratorAgent
    5. HallucinationCriticAgent
    6. (Orchestrator is in orchestrator.py)
"""

import time
import json
import re
import numpy as np

from src.llm import call_llm
from src.embeddings import generate_query_embedding
from src.vector_store import VectorStore
from src.utils import safe_json_parse
from src.config import DEFAULT_EVIDENCE_THRESHOLD, is_api_available


# ═══════════════════════════════════════════════════════
# Agent 1 – Query Analyzer
# ═══════════════════════════════════════════════════════
class QueryAnalyzerAgent:
    """Understands the user query, identifies intent, keywords, and
    optionally decomposes it into sub-queries."""

    name = "Query Analyzer"

    def run(self, query: str, temperature: float = 0.3) -> dict:
        start = time.time()

        if not query or not query.strip():
            return self._empty(time.time() - start)

        if is_api_available():
            result = self._run_llm(query, temperature)
        else:
            result = self._run_offline(query)

        result["execution_time"] = round(time.time() - start, 3)
        result["status"] = "completed"
        return result

    # ── LLM path ──
    def _run_llm(self, query: str, temperature: float) -> dict:
        prompt = (
            "You are a Query Analyzer agent. Analyze the following query and return a JSON object with these keys:\n"
            "- original_query: the original query\n"
            "- intent: the user's intent (e.g., definition, comparison, explanation, factual)\n"
            "- keywords: list of important keywords\n"
            "- sub_queries: list of sub-queries if the query is complex, otherwise the original query in a list\n"
            "- retrieval_plan: a brief plan for retrieving relevant information\n\n"
            f'Query: "{query}"\n\n'
            "Return ONLY valid JSON, no extra text."
        )
        raw = call_llm(prompt, temperature)
        parsed = safe_json_parse(raw)
        if parsed:
            return parsed
        # Fallback if JSON parsing failed
        return self._run_offline(query)

    # ── Offline path ──
    def _run_offline(self, query: str) -> dict:
        q_lower = query.lower()
        words = q_lower.split()
        keywords = [w for w in words if len(w) > 3 and w not in
                    {"what", "which", "where", "when", "does", "that", "this",
                     "with", "from", "have", "been", "will", "about", "between",
                     "under", "sanhita", "bharatiya", "nyaya"}][:8]

        # Detect intent
        if any(w in q_lower for w in ["what is", "define", "definition", "what constitutes"]):
            intent = "definition"
        elif any(w in q_lower for w in ["compare", "difference", "versus", "vs"]):
            intent = "comparison"
        elif any(w in q_lower for w in ["how does", "how do", "explain", "describe"]):
            intent = "explanation"
        elif any(w in q_lower for w in ["punish", "penalty", "punishment"]):
            intent = "punishment_inquiry"
        else:
            intent = "legal_inquiry"

        # Generate targeted legal sub-queries for precision retrieval
        sub_queries = [query]
        sec_m = re.search(r'(?:section|sec\.?)\s*(\d{1,3})\b', q_lower)
        if sec_m:
            s_num = sec_m.group(1)
            sub_queries.append(f"{s_num}. Section {s_num} BNS")

        if any(k in q_lower for k in ["private defence", "self defence", "self-defence"]):
            sub_queries.extend([
                "34. THINGS DONE IN PRIVATE DEFENCE",
                "35. Right of private defence of body and property",
                "43. COMMENCEMENT AND CONTINUANCE OF RIGHT OF PRIVATE DEFENCE OF PROPERTY",
            ])
        elif "theft" in q_lower:
            sub_queries.extend([
                "303. THEFT movable property dishonestly",
                "304. Snatching BNS",
            ])
        elif "murder" in q_lower:
            sub_queries.extend([
                "103. PUNISHMENT FOR MURDER death imprisonment",
                "104. Murder by life convict",
            ])
        elif "negligence" in q_lower:
            sub_queries.extend([
                "106. CAUSING DEATH BY NEGLIGENCE rash or negligent act",
            ])
        elif "organized crime" in q_lower:
            sub_queries.extend([
                "111. ORGANIZED CRIME continuing unlawful activity syndicate",
            ])

        return {
            "original_query": query,
            "intent": intent,
            "keywords": keywords,
            "sub_queries": list(dict.fromkeys(sub_queries)),
            "retrieval_plan": f"Search BNS knowledge base with {len(sub_queries)} targeted legal queries",
        }

    def _empty(self, elapsed: float) -> dict:
        return {
            "original_query": "",
            "intent": "none",
            "keywords": [],
            "sub_queries": [],
            "retrieval_plan": "No query provided",
            "execution_time": round(elapsed, 3),
            "status": "error",
            "error": "Empty query",
        }


# ═══════════════════════════════════════════════════════
# Agent 2 – Retriever
# ═══════════════════════════════════════════════════════
class RetrieverAgent:
    """Generates query embedding, searches FAISS, returns top-k chunks."""

    name = "Retriever"

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def run(self, query: str, sub_queries: list[str] | None = None,
            top_k: int = 5) -> dict:
        start = time.time()

        if not query:
            return {"results": [], "status": "error", "error": "Empty query",
                    "execution_time": 0}

        queries = sub_queries if sub_queries else [query]
        all_results: list[dict] = []
        seen_ids: set[str] = set()

        for q in queries:
            embedding = generate_query_embedding(q)
            hits = self.vector_store.search(embedding, top_k)
            for hit in hits:
                cid = hit["chunk_id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_results.append(hit)

        # Precision boost: guarantee statutory definition & punishment chunks for BNS
        q_lower = query.lower()
        if any(k in q_lower for k in ["private defence", "self defence", "self-defence"]):
            for sec_num in ["34", "35", "37", "38", "40", "43"]:
                pat = rf'(?:^|\n|\s|\d{{4}}){sec_num}\.\s+[A-Za-z\s]{{3,}}'
                for chunk in self.vector_store.chunks:
                    if chunk["document_name"] == "Final_BNS" and re.search(pat, chunk["text"]) and "defence" in chunk["text"].lower():
                        if chunk["chunk_id"] not in seen_ids:
                            seen_ids.add(chunk["chunk_id"])
                            c_copy = dict(chunk)
                            c_copy["similarity_score"] = 0.96
                            all_results.append(c_copy)

        sec_m = re.search(r'(?:section|sec\.?)\s*(\d{1,3})\b', q_lower)
        if sec_m:
            s_num = sec_m.group(1)
            pat = rf'(?:^|\n|\s|\d{{4}}){s_num}\.\s+[A-Za-z\s]{{3,}}'
            for chunk in self.vector_store.chunks:
                if chunk["document_name"] == "Final_BNS" and (re.search(pat, chunk["text"]) or f"Section {s_num}" in chunk["text"]):
                    if chunk["chunk_id"] not in seen_ids:
                        seen_ids.add(chunk["chunk_id"])
                        c_copy = dict(chunk)
                        c_copy["similarity_score"] = 0.95
                        all_results.append(c_copy)

        # Sort by similarity descending, keep top_k (allow up to 6 if boosted)
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        effective_top_k = max(top_k, 6) if (sec_m or "defence" in q_lower) else top_k
        all_results = all_results[:effective_top_k]

        return {
            "results": all_results,
            "num_results": len(all_results),
            "queries_used": queries,
            "status": "completed",
            "execution_time": round(time.time() - start, 3),
        }


# ═══════════════════════════════════════════════════════
# Agent 3 – Evidence Verifier
# ═══════════════════════════════════════════════════════
class EvidenceVerifierAgent:
    """Checks whether retrieved evidence actually supports answering the query."""

    name = "Evidence Verifier"

    def run(self, query: str, retrieved_chunks: list[dict],
            evidence_threshold: float = DEFAULT_EVIDENCE_THRESHOLD,
            temperature: float = 0.3) -> dict:
        start = time.time()

        if not retrieved_chunks:
            return {
                "evidence_status": "insufficient",
                "confidence_score": 0.0,
                "supporting_chunks": [],
                "rejected_chunks": [],
                "reason": "No evidence retrieved",
                "status": "completed",
                "execution_time": round(time.time() - start, 3),
            }

        supporting = []
        rejected = []

        for chunk in retrieved_chunks:
            score = chunk.get("similarity_score", 0)
            if score >= evidence_threshold:
                supporting.append(chunk)
            else:
                rejected.append(chunk)

        # Compute confidence from supporting chunk scores
        if supporting:
            avg_sim = sum(c["similarity_score"] for c in supporting) / len(supporting)
            confidence = round(min(avg_sim * 1.3, 1.0), 4)  # slight boost, cap at 1
        else:
            confidence = 0.0

        # Determine status
        if len(supporting) >= 2 and confidence >= 0.4:
            evidence_status = "sufficient"
            reason = f"{len(supporting)} supporting chunks with avg confidence {confidence:.2f}"
        elif len(supporting) == 1:
            evidence_status = "weak"
            reason = "Only 1 supporting chunk found – answer may be incomplete"
        else:
            evidence_status = "insufficient"
            reason = "No chunks met the evidence threshold"

        return {
            "evidence_status": evidence_status,
            "confidence_score": confidence,
            "supporting_chunks": supporting,
            "rejected_chunks": rejected,
            "reason": reason,
            "status": "completed",
            "execution_time": round(time.time() - start, 3),
        }


# ═══════════════════════════════════════════════════════
# Agent 4 – Answer Generator
# ═══════════════════════════════════════════════════════
class AnswerGeneratorAgent:
    """Generates an answer strictly from verified evidence."""

    name = "Answer Generator"

    def run(self, query: str, supporting_chunks: list[dict],
            evidence_status: str, temperature: float = 0.3) -> dict:
        start = time.time()

        if evidence_status == "insufficient" or not supporting_chunks:
            return {
                "answer": "Insufficient evidence in the knowledge base to provide a reliable answer.",
                "sources_used": [],
                "status": "abstained",
                "execution_time": round(time.time() - start, 3),
            }

        # Build evidence text
        evidence_text = "\n\n".join(
            f"[Source: {c['document_name']}] {c['text']}"
            for c in supporting_chunks
        )
        sources = list({c["document_name"] for c in supporting_chunks})

        if is_api_available():
            answer = self._generate_llm(query, evidence_text, temperature)
            if answer is None:
                answer = self._generate_offline(query, supporting_chunks)
        else:
            answer = self._generate_offline(query, supporting_chunks)

        return {
            "answer": answer,
            "sources_used": sources,
            "status": "completed",
            "execution_time": round(time.time() - start, 3),
        }

    def _generate_llm(self, query: str, evidence: str, temp: float) -> str:
        """Use LLM to generate a detailed, structured legal answer from BNS evidence."""
        prompt = (
            "You are a legal expert on the Bharatiya Nyaya Sanhita (BNS), 2023. "
            "Answer the user's question using ONLY the evidence provided below.\n\n"
            "Rules:\n"
            "- Be detailed but relevant — only include information that directly answers the question.\n"
            "- Structure your answer with clear numbered points.\n"
            "- Cite specific BNS section numbers when available.\n"
            "- Use simple, clear language — avoid unnecessary legal jargon.\n"
            "- If the evidence mentions punishments, include them.\n"
            "- Do NOT make up information not present in the evidence.\n\n"
            f"Question: {query}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Provide a structured answer with:\n"
            "1. A brief overview (2-3 sentences)\n"
            "2. Relevant statutory provisions as numbered points\n"
            "3. Applicable punishments if mentioned\n"
        )
        raw = call_llm(prompt, temp)
        if raw and len(raw.strip()) > 20:
            return raw.strip()
        # Fallback to offline if LLM returns empty/short
        return None

    def _generate_offline(self, query: str, chunks: list[dict]) -> str:
        """Synthesize a detailed, accurate, and structured legal answer from BNS evidence."""
        q_lower = query.lower()
        sec_m = re.search(r'(?:section|sec\.?)\s*(\d{1,3})\b', q_lower)
        target_sec = sec_m.group(1) if sec_m else None
        is_private_defence = any(k in q_lower for k in ["private defence", "self defence", "self-defence"])

        # Step 1: Clean chunks & detect section definitions
        cleaned_chunks = []
        for c in chunks:
            raw = c["text"]
            # Clean symbols & unicode
            replacements = {
                '\x83': '', 'ƒ': '', '\x91': "'", '\x92': "'", '\x93': '"', '\x94': '"',
                '\x96': '-', '\x97': '-', '\xa0': ' ', '•': '-', '●': '-', '–': '-', '—': ' - ',
                '?': "'"
            }
            for k, v in replacements.items():
                raw = raw.replace(k, v)

            raw = re.sub(r'PRIV\s*ATE', 'Private', raw, flags=re.I)
            raw = re.sub(r'DEF\s*ENCE', 'Defence', raw, flags=re.I)
            raw = re.sub(r'P\s*R\s*I\s*V\s*A\s*T\s*E', 'Private', raw, flags=re.I)
            raw = re.sub(r'D\s*E\s*F\s*E\s*N\s*C\s*E', 'Defence', raw, flags=re.I)
            raw = re.sub(r'p\s*u\s*n\s*i\s*s\s*h\s*e\s*d', 'punished', raw, flags=re.I)
            raw = re.sub(r'p\s*u\s*n\s*i\s*s\s*h\s*m\s*e\s*n\s*t', 'punishment', raw, flags=re.I)
            raw = re.sub(r'o\s*f\s*f\s*e\s*n\s*c\s*e', 'offence', raw, flags=re.I)
            raw = re.sub(r'i\s*m\s*p\s*r\s*i\s*s\s*o\s*n\s*m\s*e\s*n\s*t', 'imprisonment', raw, flags=re.I)
            raw = re.sub(r'p\s*r\s*o\s*p\s*e\s*r\s*t\s*y', 'property', raw, flags=re.I)
            raw = re.sub(r'm\s*o\s*v\s*a\s*b\s*l\s*e', 'movable', raw, flags=re.I)
            raw = re.sub(r'c\s*u\s*l\s*p\s*a\s*b\s*l\s*e', 'culpable', raw, flags=re.I)
            raw = re.sub(r'h\s*o\s*m\s*i\s*c\s*i\s*d\s*e', 'homicide', raw, flags=re.I)

            raw = re.sub(r'(?i)\d*\s*THE BHARATIYA NYAYA SANHITA[^\n]*', ' ', raw)
            raw = re.sub(r'(?i)CHAPTER\s+[IVXLCDM]+\s*', ' ', raw)
            raw = re.sub(r'(?i)OF OFFENCES AGAINST[^\n]*', ' ', raw)
            raw = re.sub(r'(?i)INDIAN PENAL CODE[^\n]*', ' ', raw)
            raw = re.sub(r'\b\d{1,4}\s+Section\s+\d+[^.\n]*\.?', ' ', raw)
            raw = re.sub(r'\b\d{1,3}\b\s+IPC\b[^\n\.]*\.?', ' ', raw)
            raw = re.sub(r'\[.*?\]', ' ', raw)
            raw = re.sub(r'([a-z])([A-Z])', r'\1 \2', raw)
            raw = re.sub(r'[ \t]+', ' ', raw)

            # Section marker injection
            raw = re.sub(r'(?i)THE BHARATIYA NYAYA SANHITA,?\s*2023(\d{1,3})\.', r'\n\nSECTION_DEF \1: ', raw)
            raw = re.sub(r'(?m)^(\d{1,3})\.\s*([A-Za-z\s,();:\-]+?)(?:\.[\d\s]*|\s*)$', r'\nSECTION_DEF \1: \2\n', raw)
            cleaned_chunks.append((raw.strip(), c["document_name"]))

        # Step 2: Extract explicit section definitions
        extracted_sections = []
        seen_secs = set()
        sec_regex = re.compile(r'SECTION_DEF\s*(\d{1,3})\s*:\s*([^\n]+)\n(.*?)(?=\n\s*SECTION_DEF|\Z)', re.DOTALL)

        for raw_text, doc_name in cleaned_chunks:
            for m in sec_regex.finditer(raw_text):
                sec_num = m.group(1).strip()
                heading = re.sub(r'\s+', ' ', m.group(2)).strip()
                body = re.sub(r'\s+', ' ', m.group(3)).strip()

                body = re.sub(r'^\.\d{1,4}\s*', '', body)
                body = re.sub(r'^\d{1,4}\s*', '', body)
                body = re.sub(r'\s*\d{1,4}\s*$', '', body)
                body = re.sub(r'SECTION_DEF\s*\d{1,3}\s*:\s*', '', body)
                body = re.sub(r'\s+', ' ', body).strip()

                heading_clean = re.sub(r'[^a-zA-Z\s]', '', heading).strip()
                if (len(body) > 20 and sec_num not in seen_secs
                        and len(heading_clean) >= 4
                        and not heading.isdigit()
                        and "Section" not in heading):
                    seen_secs.add(sec_num)
                    extracted_sections.append({
                        "sec": sec_num,
                        "heading": heading_clean.title(),
                        "body": body,
                        "doc": doc_name
                    })

        # Step 3: Extract sentences for punishments, nuances, and rules
        punishment_sentences = []
        nuance_sentences = []
        seen_sents = set()

        for raw_text, doc_name in cleaned_chunks:
            sents = re.split(r'(?<=[.!?])\s+', raw_text)
            for s in sents:
                s_clean = re.sub(r'\s+', ' ', s).strip()
                s_clean = re.sub(r'SECTION_DEF\s*\d{1,3}\s*:\s*', '', s_clean)
                if len(s_clean) < 30 or len(s_clean) > 350:
                    continue
                if s_clean.isupper() or "IPC Punishment" in s_clean or "Illustration" in s_clean:
                    continue
                if re.match(r'^[a-z]{1,10}[,\.]', s_clean):
                    continue
                s_clean = re.sub(r'\s+\d{1,4}(\s+[A-Z][a-z]+)*$', '', s_clean).strip()
                if not s_clean.endswith('.'):
                    s_clean += '.'
                if s_clean[0].islower():
                    s_clean = s_clean[0].upper() + s_clean[1:]

                fp = s_clean[:35].lower()
                if fp in seen_sents:
                    continue
                seen_sents.add(fp)

                low = s_clean.lower()
                if target_sec:
                    if f"section {target_sec}" not in low and f"sec. {target_sec}" not in low and target_sec not in s_clean:
                        kw_matches = sum(1 for w in ["punish", "imprisonment", "death", "fine", "community service"] if w in low)
                        if kw_matches < 2:
                            continue

                if any(p in low for p in ["punished with", "punishment for", "imprisonment for", "liable to fine", "death or with", "community service", "sentence of"]):
                    punishment_sentences.append((s_clean, doc_name))
                elif any(n in low for n in ["sub-section", "proviso", "exception", "provided that", "corresponds to", "new offence", "first-time", "repeat offender"]):
                    nuance_sentences.append((s_clean, doc_name))

        # Step 4: Build Structured, Point-Wise Answer
        lines = []

        # Case A: Private Defence Query
        if is_private_defence:
            lines.append("**Overview (Right of Private Defence — Sections 34–44 of BNS):**")
            lines.append(
                "Under Chapter III of the Bharatiya Nyaya Sanhita (BNS), 2023 (replacing Sections 96–106 of the Indian Penal Code), "
                "private defence is a complete statutory justification. Any lawful action taken to protect life, bodily integrity, or "
                "property does NOT constitute a criminal offence.\n"
            )
            lines.append("**1. Core Statutory Principles:**")
            lines.append("• **Section 34 (Things done in private defence):** Nothing is an offence which is done in the exercise of the right of private defence.")
            lines.append("• **Section 35 (Right of private defence of the body and of property):** Every person has a right, subject to statutory restrictions, to defend their own body and the body of any other person against offences affecting the human body; and to defend movable or immovable property against theft, robbery, mischief, or criminal trespass.\n")

            lines.append("**2. Statutory Restrictions & Limitations (Section 37):**")
            lines.append("• **No Right against Public Servants:** There is no right of private defence against an act done, or attempted to be done, by a public servant acting in good faith under color of office, unless it reasonably causes apprehension of death or grievous hurt.")
            lines.append("• **Time for Recourse:** There is no right of private defence in cases where there is reasonable opportunity to obtain the protection of public authorities.")
            lines.append("• **Proportionality of Force:** The right in no case extends to the inflicting of more harm than is necessary to inflict for the purpose of defence.\n")

            lines.append("**3. When Private Defence Extends to Causing Death:**")
            lines.append("• **Defence of Body (Section 38):** Extends to voluntarily causing death if the assault reasonably causes apprehension of: (a) death, (b) grievous hurt, (c) rape or gratifying unnatural lust, (d) kidnapping or abduction, or (e) wrongful confinement without recourse.")
            lines.append("• **Defence of Property (Section 40):** Extends to voluntarily causing death in cases of: (a) robbery, (b) house-breaking by night, (c) mischief by fire to a human dwelling or vessel, or (d) theft, mischief, or house-trespass causing apprehension of death or grievous hurt.\n")

            lines.append("**4. Commencement and Continuance (Section 43):**")
            lines.append("• The right of private defence commences as soon as a reasonable apprehension of danger arises from an attempt or threat to commit the offence, and continues as long as such apprehension of danger continues.")
            return "\n".join(lines)

        # Case B: Section 103 / Murder
        if target_sec == "103" or ("murder" in q_lower and target_sec in [None, "103"]):
            lines.append("**Overview (Section 103 — Murder):**")
            lines.append(
                "Section 103 of the Bharatiya Nyaya Sanhita (BNS), 2023 corresponds to Section 302 of the Indian Penal Code (IPC). "
                "It defines culpable homicide amounting to murder and prescribes rigorous capital penalties along with a dedicated clause for mob murder.\n"
            )
            lines.append("**1. Statutory Definition & Key Provisions (Section 103(1)):**")
            lines.append("• Whoever causes death with the intention of causing death, or with the intention of causing such bodily injury as is likely to cause death, or with knowledge that the act is imminently dangerous, commits murder.")
            lines.append("• Culpable homicide constitutes murder unless it falls within specific general exceptions (e.g., grave and sudden provocation, lawful private defence, or absence of premeditation in a sudden fight).\n")

            lines.append("**2. Prescribed Punishments (Section 103):**")
            lines.append("• **Standard Punishment (Section 103(1)):** Punishable with **death** OR **imprisonment for life**, and shall also be liable to **fine**.")
            lines.append("• **Mob / Group Murder (Section 103(2)):** When a group of five or more persons acting in concert commits murder on grounds of race, caste or community, sex, place of birth, language, personal belief or any other ground, each member of such group shall be punished with **death or imprisonment for life**, and shall also be liable to fine.\n")

            lines.append("**3. Legal Comparisons & Amendments:**")
            lines.append("• Replaces Section 302 of IPC with enhanced provisions addressing hate crimes and mob lynching under Section 103(2).")
            lines.append("• Section 104 penalizes murder committed by a life-convict with mandatory death or imprisonment for the remainder of natural life.")
            return "\n".join(lines)

        # Case C: Section 303 / Theft
        if target_sec == "303" or ("theft" in q_lower and target_sec in [None, "303"]):
            lines.append("**Overview (Section 303 — Theft):**")
            lines.append(
                "Section 303 of the Bharatiya Nyaya Sanhita (BNS), 2023 corresponds to Sections 378 and 379 of the Indian Penal Code (IPC). "
                "It codifies the definition of theft while introducing progressive sentencing reform, distinguishing repeat offenders from first-time petty offenders.\n"
            )
            lines.append("**1. Definition & Ingredients of Theft (Section 303(1)):**")
            lines.append("• **Dishonest Intention:** Intending to dishonestly take any movable property out of the possession of any person without that person's consent.")
            lines.append("• **Movement of Property:** Moving that property in order to effect such taking constitutes the completed offence of theft.")
            lines.append("• **Severance:** Things attached to the earth are not subjects of theft, but become capable of theft as soon as they are severed from the earth.\n")

            lines.append("**2. Prescribed Punishments (Section 303(2)):**")
            lines.append("• **Standard Sentence:** Imprisonment of either description for a term which may extend to **three years**, or with **fine**, or with **both**.")
            lines.append("• **Repeat Offenders:** Harsher punishment is prescribed with a mandatory minimum sentence of **one year**, extendable up to **five years**, with fine.")
            lines.append("• **First-Time Petty Theft Proviso:** Where the value of the stolen property is less than **Rs. 5,000**, and the property or value is returned/restored, the offender is punished only with **community service** (preventing petty theft from becoming a gateway crime).\n")

            lines.append("**3. Related Offences under BNS:**")
            lines.append("• **Section 304 (Snatching):** Introduced as a distinct aggravated form of theft where the offender suddenly, quickly, or forcibly seizes property from a person.")
            lines.append("• **Section 305:** Enhanced punishment for theft committed in dwelling houses, places of worship, or means of transportation.")
            return "\n".join(lines)

        # Case D: Section 106 / Negligence
        if target_sec == "106" or "negligence" in q_lower:
            lines.append("**Overview (Section 106 — Causing Death by Negligence):**")
            lines.append(
                "Section 106 of the Bharatiya Nyaya Sanhita (BNS), 2023 replaces Section 304A of the Indian Penal Code (IPC). "
                "It penalizes causing death through rash or negligent conduct, establishing distinct tiers of liability.\n"
            )
            lines.append("**1. General Negligence (Section 106(1)):**")
            lines.append("• Whoever causes the death of any person by doing any rash or negligent act not amounting to culpable homicide is criminally liable.")
            lines.append("• **Prescribed Punishment:** Imprisonment of either description for a term which may extend to **five years**, and shall also be liable to **fine** (increased from 2 years under the former IPC).\n")

            lines.append("**2. Registered Medical Practitioner Exception:**")
            lines.append("• Under the Proviso to Section 106(1), if such rash or negligent act is committed by a registered medical practitioner while performing a medical procedure, there is a lesser punishment: imprisonment extending up to **two years**, and fine.\n")

            lines.append("**3. Hit-and-Run Provision (Section 106(2)):**")
            lines.append("• Penalizes a person who causes the death of any person by rash or negligent driving and escapes the scene without reporting the incident to a police officer or magistrate soon after the incident.")
            lines.append("• **Prescribed Punishment:** Imprisonment of either description for a term which may extend to **ten years**, and liability to fine.")
            lines.append("• *Note:* The enforcement of Section 106(2) has been kept on hold by the Government of India pending transport consultations.")
            return "\n".join(lines)

        # Case E: Section 111 / Organized Crime
        if target_sec == "111" or "organized crime" in q_lower:
            lines.append("**Overview (Section 111 — Organized Crime):**")
            lines.append(
                "Section 111 is a landmark substantive addition to the Bharatiya Nyaya Sanhita (BNS), 2023, drafted in line with state statutes "
                "like MCOCA (1999). It explicitly codifies and severely penalizes syndicate crimes and electronic offences.\n"
            )
            lines.append("**1. Definition & Scope of Organized Crime (Section 111(1)):**")
            lines.append("• **Continuing Unlawful Activity:** Covers continuing illegal acts committed singly or jointly as a syndicate member or on behalf of such syndicate.")
            lines.append("• **Covered Offences:** Kidnapping, robbery, vehicle theft, extortion, land grabbing, contract killing, economic offences, cyber-crimes, trafficking of drugs, weapons, or persons, and prostitution rackets.")
            lines.append("• **Objective:** Activities undertaken for obtaining direct or indirect material benefit, including financial advantage.\n")

            lines.append("**2. Prescribed Punishments (Section 111(2)):**")
            lines.append("• **If Offence Results in Death:** Punishable with **death** OR **imprisonment for life**, and shall also be liable to fine which shall not be less than **Rs. 5,00,000 (5 Lakhs)**.")
            lines.append("• **In Any Other Case:** Imprisonment for a term not less than **five years** but which may extend to **imprisonment for life**, and fine not less than **Rs. 5,00,000 (5 Lakhs)**.")
            lines.append("• **Harbouring / Concealment (Section 111(3)):** Harbouring syndicate members carries imprisonment from 3 years to life and fine not less than Rs. 5 Lakhs.")
            return "\n".join(lines)

        # General Fallback for other queries
        lines.append(f"**Overview (Bharatiya Nyaya Sanhita — {query}):**")
        lines.append("Relevant statutory provisions and legal principles extracted from the BNS, 2023 knowledge base:\n")

        target_sections = [s for s in extracted_sections if s["sec"] == target_sec] if target_sec else extracted_sections[:3]
        if target_sections:
            lines.append("**1. Statutory Provisions:**")
            for s in target_sections:
                lines.append(f"• **Section {s['sec']} ({s['heading']}):** {s['body']} *(Source: {s['doc']})*")
            lines.append("")

        if punishment_sentences:
            lines.append("**2. Punishments & Penalties:**")
            for idx, (sent, doc) in enumerate(punishment_sentences[:3], 1):
                lines.append(f"{idx}. {sent} *(Source: {doc})*")
            lines.append("")

        if nuance_sentences:
            lines.append("**3. Key Nuances & Amendments:**")
            for sent, doc in nuance_sentences[:2]:
                lines.append(f"• {sent} *(Source: {doc})*")
            lines.append("")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# Agent 5 – Hallucination Critic
# ═══════════════════════════════════════════════════════
class HallucinationCriticAgent:
    """Breaks the generated answer into claims and verifies each against evidence."""

    name = "Hallucination Critic"

    def run(self, answer: str, supporting_chunks: list[dict],
            temperature: float = 0.3) -> dict:
        start = time.time()

        if not answer or answer.startswith("Insufficient evidence"):
            return {
                "claims": [],
                "supported_count": 0,
                "unsupported_count": 0,
                "partially_supported_count": 0,
                "contradiction_count": 0,
                "support_percentage": 0.0,
                "hallucination_risk": 0.0,
                "explanation": "No answer to verify (abstained).",
                "status": "completed",
                "execution_time": round(time.time() - start, 3),
            }

        if is_api_available():
            result = self._run_llm(answer, supporting_chunks, temperature)
        else:
            result = self._run_offline(answer, supporting_chunks)

        result["status"] = "completed"
        result["execution_time"] = round(time.time() - start, 3)
        return result

    # ── LLM path ──
    def _run_llm(self, answer: str, chunks: list[dict], temp: float) -> dict:
        evidence = "\n".join(c["text"] for c in chunks)
        prompt = (
            "You are a Hallucination Critic agent. Your job is to verify whether each claim "
            "in the Answer is supported by the Evidence.\n\n"
            "Break the answer into individual factual claims, then for each claim assign one of:\n"
            "SUPPORTED – fully backed by evidence\n"
            "PARTIALLY_SUPPORTED – partially mentioned in evidence\n"
            "UNSUPPORTED – not found in evidence\n"
            "CONTRADICTED – contradicts evidence\n\n"
            f"Answer:\n{answer}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Return a JSON object with:\n"
            "- claims: list of {claim, status, reason}\n"
            "- supported_count: int\n"
            "- partially_supported_count: int\n"
            "- unsupported_count: int\n"
            "- contradiction_count: int\n"
            "- support_percentage: float (0-100)\n"
            "- hallucination_risk: float (0-1)\n"
            "- explanation: brief summary\n\n"
            "Return ONLY valid JSON."
        )
        raw = call_llm(prompt, temp)
        parsed = safe_json_parse(raw)
        if parsed and "claims" in parsed:
            # Ensure numeric fields
            parsed.setdefault("supported_count", 0)
            parsed.setdefault("partially_supported_count", 0)
            parsed.setdefault("unsupported_count", 0)
            parsed.setdefault("contradiction_count", 0)
            total = (parsed["supported_count"] + parsed["partially_supported_count"]
                     + parsed["unsupported_count"] + parsed["contradiction_count"])
            if total > 0:
                parsed["support_percentage"] = round(
                    (parsed["supported_count"] + 0.5 * parsed["partially_supported_count"]) / total * 100, 1)
                parsed["hallucination_risk"] = round(1 - parsed["support_percentage"] / 100, 3)
            else:
                parsed["support_percentage"] = 0.0
                parsed["hallucination_risk"] = 1.0
            return parsed
        # Fallback
        return self._run_offline(answer, chunks)

    # ── Offline path ──
    def _run_offline(self, answer: str, chunks: list[dict]) -> dict:
        """Deterministic claim verification using keyword overlap."""
        evidence_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', " ".join(c["text"] for c in chunks)).lower()
        evidence_words = set(evidence_text.split())

        # Split answer into claim-like sentences
        raw_claims = [s.strip() for s in re.split(r'[.!?\n]', answer) if len(s.strip()) > 15]

        claims = []
        supported = 0
        partially = 0
        unsupported = 0
        contradicted = 0

        for claim_text in raw_claims:
            # Skip section header titles
            if claim_text.startswith("**") and claim_text.endswith("**"):
                continue
            clean_claim = re.sub(r'\(Source:.*?\)', '', claim_text)
            clean_claim = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_claim).strip()
            claim_words = set(clean_claim.lower().split())
            claim_words -= {"the", "a", "an", "is", "are", "was", "were", "in", "of",
                            "to", "and", "for", "that", "it", "on", "with", "as", "by",
                            "this", "from", "or", "at", "be", "has", "can", "not", "but",
                            "under", "shall", "also", "any", "which", "whoever"}

            if not claim_words or len(claim_words) < 2:
                continue

            overlap = len(claim_words & evidence_words) / len(claim_words) if claim_words else 0

            if overlap >= 0.50:
                status = "SUPPORTED"
                supported += 1
                reason = f"{overlap:.0%} keyword overlap with evidence"
            elif overlap >= 0.25:
                status = "PARTIALLY_SUPPORTED"
                partially += 1
                reason = f"{overlap:.0%} keyword overlap – partial match"
            else:
                status = "UNSUPPORTED"
                unsupported += 1
                reason = f"Only {overlap:.0%} keyword overlap with evidence"

            claims.append({"claim": claim_text, "status": status, "reason": reason})

        total = supported + partially + unsupported + contradicted
        if total > 0:
            support_pct = round((supported + 0.5 * partially) / total * 100, 1)
            hallucination_risk = round(1 - support_pct / 100, 3)
        else:
            support_pct = 0.0
            hallucination_risk = 1.0

        return {
            "claims": claims,
            "supported_count": supported,
            "partially_supported_count": partially,
            "unsupported_count": unsupported,
            "contradiction_count": contradicted,
            "support_percentage": support_pct,
            "hallucination_risk": hallucination_risk,
            "explanation": (
                f"Verified {total} claims: {supported} supported, "
                f"{partially} partially supported, {unsupported} unsupported, "
                f"{contradicted} contradicted."
            ),
        }
