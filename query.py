"""
query.py — Intent extraction + RAG answer generation.

Flow:
  1. LLM extracts brand + device_type from user query.
  2. retriever.py fetches only the relevant FAISS chunks (filtered by brand).
  3. LLM generates a structured, step-by-step answer using ONLY those chunks.
  4. Returns answer + source filenames + metadata for the diagnostic report.
"""

import json
from langchain_community.llms import Ollama          # swap → ChatOpenAI if preferred
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import retrieve

# ── Prompts ────────────────────────────────────────────────────────────────────

INTENT_PROMPT = PromptTemplate.from_template("""
Extract the networking device brand and device type from the user message below.
Return ONLY a JSON object with keys "brand" and "device_type".
Use lowercase. If unknown, use "unknown".

Examples:
  "my cisco switch port is blinking amber" → {{"brand": "cisco", "device_type": "switch"}}
  "the router won't connect to internet"   → {{"brand": "unknown", "device_type": "router"}}
  "huawei ONU showing red light"           → {{"brand": "huawei", "device_type": "onu"}}

Message: {query}
JSON:""")

RAG_PROMPT = PromptTemplate.from_template("""
You are a network troubleshooting assistant helping non-technical staff.
Use ONLY the context provided below to answer. Do not use outside knowledge.
If the answer is not in the context, say "I could not find this in the manuals."

Keep your answer:
- Simple and non-technical
- Numbered step-by-step where possible
- Under 200 words

Context:
{context}

User question: {question}

Answer:""")

# ── LLM setup ─────────────────────────────────────────────────────────────────
# Change to ChatOpenAI(model="gpt-4o") or ChatAnthropic if preferred
llm = Ollama(model="llama3")


# ── Functions ─────────────────────────────────────────────────────────────────

def extract_intent(query: str) -> dict:
    """Use LLM to pull brand + device_type out of the user's message."""
    chain = INTENT_PROMPT | llm | StrOutputParser()
    raw   = chain.invoke({"query": query}).strip()

    # Strip markdown fences if model wraps JSON
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
        return {
            "brand":       result.get("brand", "unknown").lower(),
            "device_type": result.get("device_type", "unknown").lower(),
        }
    except json.JSONDecodeError:
        print(f"[warn] intent parse failed, raw output: {raw!r}")
        return {"brand": "unknown", "device_type": "unknown"}


def ask(question: str) -> dict:
    """
    Main entry point. Returns a dict with:
      - answer       : step-by-step troubleshooting answer
      - sources      : list of source filenames used
      - brand        : detected brand
      - device_type  : detected device type
      - chunks_used  : number of chunks injected into the prompt
    """
    # Step 1: extract intent
    intent      = extract_intent(question)
    brand       = intent["brand"]
    device_type = intent["device_type"]
    print(f"[intent] brand={brand}, device_type={device_type}")

    # Step 2: retrieve relevant chunks from FAISS
    docs = retrieve(question, brand=brand, device_type=device_type)

    if not docs:
        return {
            "answer":      "No relevant documentation found. Please contact IT support.",
            "sources":     [],
            "brand":       brand,
            "device_type": device_type,
            "chunks_used": 0,
        }

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources = list({doc.metadata.get("source_file", "unknown") for doc in docs})

    # Step 3: generate answer using only the retrieved context
    chain  = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer":      answer.strip(),
        "sources":     sources,
        "brand":       brand,
        "device_type": device_type,
        "chunks_used": len(docs),
    }


# ── CLI usage ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "My router keeps disconnecting every few minutes"
    result   = ask(question)

    print(f"\n{'='*60}")
    print(f"Brand: {result['brand']}  |  Device: {result['device_type']}")
    print(f"Chunks used: {result['chunks_used']}  |  Sources: {', '.join(result['sources'])}")
    print(f"{'='*60}")
    print(result["answer"])
