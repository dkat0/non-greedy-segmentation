from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tokenizer import TextSegmenter

# Test Example:

# Initialize values
text = "How many rs are there in strawberry"
k = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can manually pass in a model/tokenizer if you have already created them
precreated_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B").to(device)
precreated_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

print("Starting")
segmenter = TextSegmenter(k=k, model=precreated_model, tokenizer=precreated_tokenizer)
default_tokenization, default_perplexity = segmenter.get_default_tokenization_perplexity(text)
print(f"Default Tokenization: {default_tokenization}")
print(f"Default Perplexity: {default_perplexity}")

# Compute and print results
segs = segmenter.compute_segs(text)
best_seg = segs[-1][0]
print(f"Best Seg: {best_seg[0]}")
print(f"Best Perplexity: {best_seg[1]}")
print(f"Runner Ups (Perplexity): {[seg[1] for seg in segs[-1][1:]]}")