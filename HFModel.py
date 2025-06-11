import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

class HFModel():
    def __init__(self, hf_token: str, model_name: str, temperature: float, auto_login: bool = True):
        self.max_new_tokens = 512
        self.model_name = model_name
        self.temperature = temperature
        
        if auto_login:
            self._login_to_hf(hf_token)
        
        self.tokenizer = self._load_tokenizer()
        self.quantization_config = self._set_quantization()
        self.model = self._load_model()
        self.pipe = self._set_text_gen_pipeline()
        self.hf_pipeline = self._set_hf_pipeline()

    def _login_to_hf(self, hf_token: str):
        try:
            login(hf_token)
            print("Successfully login to Hugging Face.")
        except Exception as e:
            print(f"Warning: HF login failed: {e}")
            print("Continuing without authentication - some models may not be accessible.")
    
    def _load_tokenizer(self):
        """Load and configure the tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer for {self.model_name} is loaded.")
        return tokenizer
    
    def _set_quantization(self):
        """Configure 4-bit quantization settings"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        print("The quantization config is created.")
        return quantization_config
    
    def _load_model(self):
        """Load the model with quantization"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=self.quantization_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print(f"Model {self.model_name} is loaded.")
        return model
        
    def _set_text_gen_pipeline(self):
        """Create text generation pipeline with proper sampling settings"""
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=self.temperature, # do_sample must set True to use temperature
            do_sample=True,
            return_full_text=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id
        )
        print("The text generation pipeline is created.")
        return pipe

    def _set_hf_pipeline(self):
        """Create LangChain-compatible pipeline wrapper"""
        hf_pipe = HuggingFacePipeline(
            pipeline=self.pipe,
            model_kwargs={
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True
            }
        )
        print("The Hugging Face pipeline is created.")
        return hf_pipe
        
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "vocab_size": len(self.tokenizer),
            "max_new_tokens": self.max_new_tokens,
            "model_size": sum(p.numel() for p in self.model.parameters()),
            "quantized": True,
            "device": next(self.model.parameters()).device
        }