import hashlib
import requests
from pathlib import Path
from corpus.sources import BRAND_SOURCES
from config import DOCS_DIR

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Bot/1.0)"}


def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def download_brand_pdfs():
    """Download all PDFs from BRAND_SOURCES into ./docs/<brand>/ folders."""
    for brand, urls in BRAND_SOURCES.items():
        if not urls:
            print(f"[skip] {brand} — no URLs configured")
            continue

        brand_dir = Path(DOCS_DIR) / brand
        brand_dir.mkdir(parents=True, exist_ok=True)

        for url in urls:
            filename = url.split("/")[-1].split("?")[0] or "doc.pdf"
            if not filename.endswith((".pdf", ".html", ".htm", ".md")):
                filename += ".pdf"
            dest = brand_dir / filename

            if dest.exists():
                print(f"[skip] {brand}/{filename} already downloaded")
                continue

            try:
                print(f"[download] {brand}/{filename} ...")
                r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                print(f"[ok] saved → {dest}")
            except Exception as e:
                print(f"[fail] {url}\n       reason: {e}")


if __name__ == "__main__":
    download_brand_pdfs()
