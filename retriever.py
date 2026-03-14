"""
retriever.py — Dynamic FAISS retrieval with metadata filtering.

FAISS does not support WHERE-clause pre-filtering like ChromaDB.
Strategy:
  1. Fetch TOP_K * FILTER_MULTIPLIER candidates from FAISS (oversample).
  2. Post-filter by brand (and optionally device_type) using stored metadata.
  3. Return the top TOP_K results after filtering.
  4. If filtered results < TOP_K, fall back to unfiltered top-K.

This gives brand/device-specific answers without loading the entire corpus.
"""

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import FAISS_PATH, COLLECTION_NAME, TOP_K, EMBED_MODEL

FILTER_MULTIPLIER = 10   # Oversample factor before post-filtering

_embeddings  = None
_faiss_store = None


def _load_store():
    global _embeddings, _faiss_store
    if _faiss_store is not None:
        return _faiss_store

    index_file = Path(FAISS_PATH) / f"{COLLECTION_NAME}.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_PATH}. Run ingest.py first."
        )

    _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    _faiss_store = FAISS.load_local(
        FAISS_PATH,
        _embeddings,
        index_name=COLLECTION_NAME,
        allow_dangerous_deserialization=True,
    )
    return _faiss_store


def retrieve(query: str, brand: str = None, device_type: str = None):
    """
    Retrieve the most relevant chunks for a query, filtered by brand/device_type.

    Args:
        query:       Natural language query from the user.
        brand:       Detected brand (e.g. "cisco"). None = no filter.
        device_type: Detected device type (e.g. "router"). None = no filter.

    Returns:
        List of LangChain Document objects (up to TOP_K).
    """
    store = _load_store()

    apply_filter = brand and brand.lower() not in ("unknown", "")

    if apply_filter:
        # Oversample, then post-filter by metadata
        candidates = store.similarity_search(
            query, k=TOP_K * FILTER_MULTIPLIER
        )

        filtered = [
            doc for doc in candidates
            if doc.metadata.get("brand", "").lower() == brand.lower()
        ]

        # Optionally narrow further by device_type
        if device_type and device_type.lower() not in ("unknown", ""):
            type_filtered = [
                doc for doc in filtered
                if device_type.lower() in doc.metadata.get("device_types", "").lower()
            ]
            if type_filtered:
                filtered = type_filtered

        if len(filtered) >= TOP_K:
            return filtered[:TOP_K]

        # Not enough brand-specific results → top up with unfiltered remainder
        seen_ids = {id(doc) for doc in filtered}
        extras   = [doc for doc in candidates if id(doc) not in seen_ids]
        return (filtered + extras)[:TOP_K]

    # No brand detected — pure similarity search across entire corpus
    return store.similarity_search(query, k=TOP_K)
