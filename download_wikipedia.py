import os
import requests
import subprocess

WIKI_DUMP_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
DUMP_FILE = "enwiki-latest-pages-articles.xml.bz2"
EXTRACTED_DIR = "wiki_extracted"


def download_wikipedia_dump():
    if not os.path.exists(DUMP_FILE):
        print(f"Downloading {DUMP_FILE}...")
        with requests.get(WIKI_DUMP_URL, stream=True) as r:
            r.raise_for_status()
            with open(DUMP_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    else:
        print(f"{DUMP_FILE} already exists.")

def extract_wikipedia_dump():
    if not os.path.exists(EXTRACTED_DIR):
        print("Extracting text with WikiExtractor...")
        subprocess.run([
            "wikiextractor", DUMP_FILE, "-o", EXTRACTED_DIR, "--json"
        ], check=True)
        print("Extraction complete.")
    else:
        print(f"{EXTRACTED_DIR} already exists.")

if __name__ == "__main__":
    download_wikipedia_dump()
    extract_wikipedia_dump()
    print("Wikipedia dump is ready for tokenization.")
