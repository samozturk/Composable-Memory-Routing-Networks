# file: cmrn/tokenizer.py
from transformers import AutoTokenizer

class CMRNTokenizer:
    """A wrapper around a Hugging Face tokenizer."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        """Loads a tokenizer from the Hugging Face Hub."""
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(hf_tokenizer)

    def __call__(self, text, **kwargs):
        """Tokenizes text."""
        return self.tokenizer(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        """Decodes token IDs back to text."""
        return self.tokenizer.decode(token_ids, **kwargs)

    def chunk_text(self, text: str, chunk_size: int = 2048):
        """Splits a long text into chunks of a specified token size."""
        tokens = self.tokenizer.encode(text)
        for i in range(0, len(tokens), chunk_size):
            yield self.tokenizer.decode(tokens[i:i + chunk_size])