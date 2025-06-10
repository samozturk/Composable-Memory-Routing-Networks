# file: examples/long_context.py
import torch
from cmrn import CMRN, CMRNConfig

# NOTE: This is a conceptual example. It demonstrates the API for handling
# long documents by processing them in chunks to update the model's memory.

def main():
    print("--- Long-Context Reasoning Example ---")
    
    # 1. Initialize model
    config = CMRNConfig(d_model=128, memory_dim=128) # Smaller for demo
    model = CMRN(config)
    model.train() # Set to train to allow memory updates
    
    # Mock tokenizer
    class MockTokenizer:
        def chunk_text(self, text, chunk_size):
            # Simulate chunking a long document
            return [f"Chunk {i+1} of the document..." for i in range(5)]
        def __call__(self, chunk, return_tensors):
            return torch.randint(0, config.vocab_size, (1, 100))

    tokenizer = MockTokenizer()

    long_document = "This is a very long document spanning over 100k tokens..."
    
    # 2. Process long document in chunks
    print("Resetting model memory...")
    model.reset_memory()
    
    print(f"\nProcessing long document in chunks...")
    chunks = tokenizer.chunk_text(long_document, chunk_size=2048)
    for i, chunk in enumerate(chunks):
        inputs = tokenizer(chunk, return_tensors='pt')
        _ = model(inputs) # Forward pass to simulate memory updates
        print(f"Processed chunk {i+1}. Memory usage: {model.memory_system.index.ntotal} vectors.")

    # 3. Ask a question about the document
    print("\nAsking a question about the full document...")
    model.eval()
    question = "What are the main conclusions drawn in this document?"
    question_inputs = tokenizer(question, return_tensors='pt')
    
    with torch.no_grad():
        # The generate method is a simple placeholder
        answer_ids = model.generate(question_inputs, max_length=30)
    
    print(f"\nQuestion: '{question}'")
    print(f"Answer: [Generated sequence of {answer_ids.shape[1]} tokens]")

if __name__ == '__main__':
    main()