import os
from pathlib import Path
from dotenv import load_dotenv

class Creds():
    def __init__(self):
        # Set .env
        load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
        
        # API Keys
        self.ollama_api_key = "ollama"
        self.google_api_key = self._get_env("GOOGLE_API_KEY")
        self.openai_api_key = self._get_env("OPENAI_API_KEY")
        self.hugging_face_token = self._get_env("HUGGINGFACE_TOKEN")
        
        # Base URL
        self.ollama_base_url = "http://localhost:11434/v1"
        self.google_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        
        print("All API Keys, Tokens, and Base URLs are loaded.")

    def _get_env(self, key: str) -> str:
        key = os.getenv(key)
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
        return key