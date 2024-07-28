from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os

# EvalPlus test:
# python codegen/generate.py --model "meta-llama/Meta-Llama-3-8B" --greedy --root res --dataset humaneval --backend hf --new_tokenization True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TextSegmenter:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", k=5, alpha=0.5, model=None, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model else AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.vocab = self.tokenizer.get_vocab()
        self.k = k
        self.perplexity_normalization_alpha = alpha

    def compute_segs(self, text):
        text = text.replace(' ', 'Ġ').replace("\n", 'Ċ').replace("\t", 'ĉ')
        n = len(text)
        segs = [[] for _ in range(n)]
        print(f"Iteration / {n}: ", end = '')
        
        for i in range(n):
            for j in range(i+1):
                token = text[j:i+1]
                if token in self.vocab:
                    if j == 0:
                        # Directly add seg if it's the full entry
                        segs[i].append(([token], self.calculate_seg_perplexity([token])))
                    else:
                        for k_prime in range(min(self.k, len(segs[j-1]))):
                            # Update segs[i] with new seg and calculate perplexity
                            new_seg = segs[j-1][k_prime][0] + [token]
                            segs[i].append((new_seg, self.calculate_seg_perplexity(new_seg)))
                    
            # Keep only top k segs
            segs[i] = sorted(segs[i], key=lambda x: x[1])[:self.k] # Make key=calculate_seg_perplexity for actual
            print(f"{i} ", end = '')
            if not segs[i]:
                print("Possible error has occurred, no segs generated")
                exit()
        print("")
        return segs

    def test_perplexity(self, seg):
        return len(seg)

    def calculate_seg_perplexity(self, seg):
        #input_ids = [vocab[token] for token in seg]
        #input_ids = tokenizer.convert_tokens_to_ids(seg)
        #input_ids = torch.tensor([input_ids])
        inputs = self.tokenizer(seg, is_split_into_words=True, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        return self.calculate_perplexity(input_ids)

    def get_default_tokenization_perplexity(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        return self.tokenizer.convert_ids_to_tokens(input_ids.tolist()[0]), self.calculate_perplexity(input_ids)

    def calculate_perplexity(self, input_ids):
        t1 = time.time()
        
        # Calculate Loss
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
        
        """
        Regular PPL Calculation:
        ppl = torch.exp(loss)
        print(ppl.item())
        """

        # Calculate Normalized PPL
        num_tokens = input_ids.shape[1]
        average_loss = loss / (num_tokens ** self.perplexity_normalization_alpha)
        normalized_ppl = torch.exp(average_loss)
        
        t2 = time.time()
        #print(f"Perplexity calculation finished, time = {t2-t1}")
        
        return normalized_ppl.item() 

    def tokenize(self, text):
        segs = self.compute_segs(text)
        best_seg = segs[-1][0]
        
        print(f"Best Seg: {best_seg[0]}")
        print(f"Best Perplexity: {best_seg[1]}")
        print(f"Runner Ups (Perplexity): {[seg[1] for seg in segs[-1][1:]]}")

        return best_seg[0]

if __name__ == "__main__":
    # Test Example:

    # Initialize values
    text = "catdog catfly"
    k = 10

    # You can manually pass in a model/tokenizer if you have already created them
    precreated_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    precreated_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

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
