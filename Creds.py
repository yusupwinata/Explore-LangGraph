import os
from dotenv import load_dotenv

class Creds():
    def __init__(self):
        # Ollama
        self.ollama_api_key = "ollama"
        self.ollama_base_url = "http://localhost:11434/v1"
        
        # Google
        self.google_api_key = self._get_goole_api_key()
        self.google_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        
        # Hugging Face
        self.hugging_face_token = self._get_hf_token()
        
        print("All API Keys and Tokens are loaded.")

    def _get_hf_token(self):
        return os.getenv("HUGGINGFACE_TOKEN", ".env not found.")
        
    def _get_goole_api_key(self):
        return os.getenv("GOOGLE_API_KEY", ".env not found.")