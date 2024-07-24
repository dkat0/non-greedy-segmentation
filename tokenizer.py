from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

def compute_segs(text, vocab, k):
    text = text.replace(' ', 'Ġ').replace("\n", 'Ċ').replace("\t", 'ĉ')
    n = len(text)
    segs = [[] for _ in range(n)]
    print(f"Iteration / {n}: ", end = '')
    
    for i in range(n):
        for j in range(i+1):
            token = text[j:i+1]
            if token in vocab:
                if j == 0:
                    # Directly add seg if it's the full entry
                    segs[i].append(([token], calculate_seg_perplexity([token])))
                else:
                    for k_prime in range(min(k, len(segs[j-1]))):
                        # Update segs[i] with new seg and calculate perplexity
                        new_seg = segs[j-1][k_prime][0] + [token]
                        segs[i].append((new_seg, calculate_seg_perplexity(new_seg)))
                
        # Keep only top k segs
        segs[i] = sorted(segs[i], key=lambda x: x[1])[:k] # Make key=calculate_seg_perplexity for actual
        print(f"{i} ", end = '')
        if not segs[i]:
            print("Possible error has occurred, no segs generated")
            exit()
    print("")
    return segs

def test_perplexity(seg):
    return len(seg)

def calculate_seg_perplexity(seg):
    #input_ids = [vocab[token] for token in seg]
    #input_ids = tokenizer.convert_tokens_to_ids(seg)
    #input_ids = torch.tensor([input_ids])
    inputs = tokenizer(seg, is_split_into_words=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    return calculate_perplexity(input_ids)

def get_default_tokenization_perplexity(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    return tokenizer.convert_ids_to_tokens(input_ids.tolist()[0]), calculate_perplexity(input_ids)

def calculate_perplexity(input_ids):
    # print("Starting perplexity calculation...")
    t1 = time.time()
    
    # Calculate Loss
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
    
    """
    Regular PPL Calculation:
    ppl = torch.exp(loss)
    print(ppl.item())
    """

    # Calculate Normalized PPL
    num_tokens = input_ids.shape[1]
    average_loss = loss / (num_tokens ** 0.5)
    normalized_ppl = torch.exp(average_loss)
    
    t2 = time.time()
    #print(f"Perplexity calculation finished, time = {t2-t1}")
    
    return normalized_ppl.item() 

# Initialize values
text = """def concatenate(strings: List[str]) -> str:
    \"\"\" Concatenate list of strings into a single string
    >>> concatenate([])
    ''
    >>> concatenate(['a', 'b', 'c'])
    'abc'
    \"\"\""""
vocab = tokenizer.get_vocab()
k = 10

default_tokenization, default_perplexity = get_default_tokenization_perplexity(text)
print(f"Default Tokenization: {default_tokenization}")
print(f"Default Perplexity: {default_perplexity}")

# Compute and print results
segs = compute_segs(text, vocab, k)
best_seg = segs[-1][0]
print(f"Best Seg: {best_seg[0]}")
print(f"Best Perplexity: {best_seg[1]}")
print(f"Runner Ups (Perplexity): {[seg[1] for seg in segs[-1][1:]]}")


