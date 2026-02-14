"""LLM wrapper for HuggingFace models"""

from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMWrapper:
    """Wrapper for HuggingFace LLM models"""
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: Optional[str] = None,
        token: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Initialize LLM wrapper
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            token: HuggingFace API token for private models
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model {model_name} on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print(f"Model loaded successfully on {self.device}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Override default max_new_tokens
            temperature: Override default temperature
            top_p: Override default top_p
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Use defaults if not provided
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Make the wrapper callable"""
        return self.generate(prompt, **kwargs)
