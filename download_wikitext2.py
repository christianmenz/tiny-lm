import os
import requests
import zipfile


# Use Hugging Face mirror (tar.gz format)
WIKITEXT2_URL = "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-v1.tgz"
TGZ_FILE = "wikitext-2-v1.tgz"
EXTRACTED_DIR = "wikitext-2"

import tarfile

def download_wikitext2():
    if not os.path.exists(TGZ_FILE):
        print(f"Downloading {TGZ_FILE}...")
        with requests.get(WIKITEXT2_URL, stream=True) as r:
            r.raise_for_status()
            with open(TGZ_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    else:
        print(f"{TGZ_FILE} already exists.")

def extract_wikitext2():
    if not os.path.exists(EXTRACTED_DIR):
        print("Extracting Wikitext-2...")
        with tarfile.open(TGZ_FILE, 'r:gz') as tar:
            tar.extractall()
        print("Extraction complete.")
    else:
        print(f"{EXTRACTED_DIR} already exists.")

if __name__ == "__main__":
    download_wikitext2()
    extract_wikitext2()
    print("Wikitext-2 dataset is ready for tokenization.")
