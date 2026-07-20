import shutil, os

CACHE_DIR = "cache"

if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)
    print(f"Deleted {CACHE_DIR}/")
else:
    print("No cache found.")
