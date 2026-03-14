FAISS_PATH      = "./faiss_db"         # Folder where FAISS index + docstore are saved
COLLECTION_NAME = "network_docs"       # Used as the index filename prefix
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 120
TOP_K           = 4                    # Chunks injected per query
DOCS_DIR        = "./docs"             # Manual PDF drop folder
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"   # Free, fully local
