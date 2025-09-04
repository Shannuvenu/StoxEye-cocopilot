import os
from dotenv import load_dotenv, find_dotenv

path = find_dotenv(".env", usecwd=True)
print("Found:", path)
load_dotenv(path, override=True)
print("Key:", os.getenv("OPENAI_API_KEY")[:10] if os.getenv("OPENAI_API_KEY") else None)
