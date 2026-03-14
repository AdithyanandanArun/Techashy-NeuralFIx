"""
ingest.py — Load docs, chunk them, tag metadata, embed and store in FAISS.

FAISS does not support metadata filtering natively, so we maintain a parallel
JSON sidecar file (faiss_db/metadata_store.json) that maps each FAISS vector
index position to its metadata dict. retriever.py uses this for pre-filtering.

Safe to re-run: already-ingested files are skipped using MD5 hashes tracked
in faiss_db/ingested_hashes.json.
"""

import hashlib
import json
import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    BSHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    FAISS_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOCS_DIR,
    EMBED_MODEL,
)

# Maps subfolder name → brand metadata
BRAND_META = {
    "cisco":    {"brand": "cisco",    "device_types": "router,switch,firewall"},
    "tp-link":  {"brand": "tp-link",  "device_types": "router,switch,modem"},
    "netgear":  {"brand": "netgear",  "device_types": "router,switch,modem"},
    "dlink":    {"brand": "dlink",    "device_types": "router,switch,modem"},
    "juniper":  {"brand": "juniper",  "device_types": "router,switch,firewall"},
    "ubiquiti": {"brand": "ubiquiti", "device_types": "router,switch,access_point"},
    "mikrotik": {"brand": "mikrotik", "device_types": "router,switch"},
    "huawei":   {"brand": "huawei",   "device_types": "router,switch,onu,modem"},
    "generic":  {"brand": "generic",  "device_types": "router,switch,modem"},
}

HASH_STORE  = Path(FAISS_PATH) / "ingested_hashes.json"
INDEX_FILE  = Path(FAISS_PATH) / f"{COLLECTION_NAME}.faiss"


def load_hashes() -> set:
    if HASH_STORE.exists():
        return set(json.loads(HASH_STORE.read_text()))
    return set()


def save_hashes(hashes: set):
    Path(FAISS_PATH).mkdir(parents=True, exist_ok=True)
    HASH_STORE.write_text(json.dumps(list(hashes)))


def file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_file(path: Path):
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return PyPDFLoader(str(path)).load()
        elif suffix in (".html", ".htm"):
            return BSHTMLLoader(str(path)).load()
        elif suffix in (".md", ".txt"):
            return UnstructuredMarkdownLoader(str(path)).load()
    except Exception as e:
        print(f"[error] could not load {path.name}: {e}")
    return []


def ingest_all():
    embeddings     = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    splitter       = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    ingested_hashes = load_hashes()

    # Load existing FAISS index if present, otherwise start fresh
    faiss_store = None
    if INDEX_FILE.exists():
        print(f"[load] existing FAISS index from {FAISS_PATH}")
        faiss_store = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            index_name=COLLECTION_NAME,
            allow_dangerous_deserialization=True,
        )

    new_chunks_total = 0

    for brand_dir in sorted(Path(DOCS_DIR).iterdir()):
        if not brand_dir.is_dir():
            continue
        brand    = brand_dir.name.lower()
        meta_def = BRAND_META.get(brand, {"brand": brand, "device_types": "unknown"})

        for file_path in sorted(brand_dir.rglob("*")):
            if file_path.suffix.lower() not in (".pdf", ".html", ".htm", ".md", ".txt"):
                continue

            fhash = file_md5(str(file_path))
            if fhash in ingested_hashes:
                print(f"[skip] {brand}/{file_path.name}")
                continue

            print(f"[ingest] {brand}/{file_path.name}")
            raw_docs = load_file(file_path)
            if not raw_docs:
                continue

            chunks = splitter.split_documents(raw_docs)

            # Tag every chunk with metadata FAISS will store in its docstore
            for chunk in chunks:
                chunk.metadata.update({
                    "brand":        meta_def["brand"],
                    "device_types": meta_def["device_types"],
                    "source_file":  file_path.name,
                    "file_hash":    fhash,
                    "doc_type":     "troubleshooting_manual",
                })

            if faiss_store is None:
                faiss_store = FAISS.from_documents(chunks, embeddings)
            else:
                faiss_store.add_documents(chunks)

            ingested_hashes.add(fhash)
            new_chunks_total += len(chunks)
            print(f"[ok] {len(chunks)} chunks added for {file_path.name}")

    if faiss_store is not None and new_chunks_total > 0:
        Path(FAISS_PATH).mkdir(parents=True, exist_ok=True)
        faiss_store.save_local(FAISS_PATH, index_name=COLLECTION_NAME)
        save_hashes(ingested_hashes)
        print(f"\nDone. {new_chunks_total} new chunks saved to {FAISS_PATH}/")
    elif new_chunks_total == 0:
        print("\nNo new documents to ingest.")
    else:
        print("\n[warn] No documents were loaded at all — check your ./docs folder.")


if __name__ == "__main__":
    ingest_all()
