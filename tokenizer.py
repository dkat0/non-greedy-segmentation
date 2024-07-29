from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os
import argparse

# Reg CLI test:
#
# Compare Default vs. Perplexity-Optimized Segmentation:
#   python tokenizer.py --text "How many rs are there in strawberry" --k 10 --alpha 0.5 --model_name "meta-llama/Meta-Llama-3-8B"
#
# Obtain Perplexity of User-Defined Segmentation:
#   python tokenizer.py --user_segmentation "How, many, rs, are, there, in, straw,berry" --alpha 0.5 --model_name "meta-llama/Meta-Llama-3-8B" 
#   -> This tests the perplexity of the segmentation ["How", " many", " rs", " are", " there", " in", " straw", "berry"]

# EvalPlus test:
#   python codegen/generate.py --model "meta-llama/Meta-Llama-3-8B" --greedy --root res --dataset humaneval --backend hf --new_tokenization True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TextSegmenter:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", k=5, alpha=0.5, model=None, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model else AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.vocab = self.tokenizer.get_vocab()
        self.k = k
        self.perplexity_normalization_alpha = alpha

    @staticmethod
    def encode_text_special(text):
        return text.replace(' ', 'Ġ').replace("\n", 'Ċ').replace("\t", 'ĉ')

    @staticmethod
    def decode_text_special(text):
        return text.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')

    def compute_segs(self, text):
        n = len(text)
        segs = [[] for _ in range(n)]
        print(f"Iteration / {n}: ", end = '')
        
        for i in range(n):
            for j in range(i+1):
                token = text[j:i+1]
                if self.encode_text_special(token) in self.vocab:
                    if j == 0:
                        # Directly add seg if it's the full entry
                        segs[i].append(([token], self.calculate_seg_perplexity([token])))
                    else:
                        for k_prime in range(len(segs[j-1]) if self.k is None else min(self.k, len(segs[j-1]))):
                            # Update segs[i] with new seg and calculate perplexity
                            new_seg = segs[j-1][k_prime][0] + [token]
                            segs[i].append((new_seg, self.calculate_seg_perplexity(new_seg)))
                    
            # Sort segments by perplexity
            segs[i] = sorted(segs[i], key=lambda x: x[1])

            # Keep only top k segs (if applicable)
            if self.k is not None:
                segs[i] = segs[i] = segs[i][:self.k]
            
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
        
        return normalized_ppl.item() 

    def tokenize(self, text):
        segs = self.compute_segs(text)
        best_seg = segs[-1][0]
        
        print(f"Best Seg: {best_seg[0]}")
        print(f"Best Perplexity: {best_seg[1]}")
        print(f"Runner Ups (Perplexity): {[seg[1] for seg in segs[-1][1:]]}")

        return best_seg[0]
    
    def seg_to_token_list(self, seg):
        input_ids = self.tokenizer(seg, is_split_into_words=True, return_tensors="pt").to(self.device)["input_ids"]
        return self.tokenizer.convert_ids_to_tokens(input_ids.tolist()[0])

    def generate_response(self, seg):
        start_token_id = self.model.config.bos_token_id # for Llama-3 models, this is 128000
        start_token = self.tokenizer.convert_ids_to_tokens(start_token_id) # for Llama-3 models, this is "<|begin_of_text|>"

        # Remove the start token if it is passed in
        if seg[0] == start_token:
            seg = seg[1:]

        seg = [self.decode_text_special(token) for token in seg]

        text = "".join(seg)
        messages = [
            {"role": "user", "content": text}
        ]

        chat_template = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        before_text, after_text = chat_template.split(text)

        before_tokens = self.tokenizer(before_text, return_tensors="pt").to(self.device)["input_ids"]
        after_tokens = self.tokenizer(after_text, return_tensors="pt").to(self.device)["input_ids"]
        text_tokens = self.tokenizer(seg, is_split_into_words=True, return_tensors="pt").to(self.device)["input_ids"]

        # Remove all start tokens, since the tokenizer will incorrectly add one to the beginning of all of these
        for i in range(2): # Since the Llama tokenizer seems to incorrectly add two start tokens when using chat templating
            if before_tokens[0, 0] == start_token_id:
                before_tokens = before_tokens[:, 1:]
        if text_tokens[0, 0] == start_token_id:
            text_tokens = text_tokens[:, 1:]
        if after_tokens[0, 0] == start_token_id:
            after_tokens = after_tokens[:, 1:]

        # Concatenate tokens, finally adding the start token only once at the beginning
        new_seg = torch.concat([torch.tensor([[start_token_id]]).to(self.device), before_tokens, text_tokens, after_tokens], dim=1)
        
        with torch.no_grad():
            outputs = self.model.generate(new_seg, attention_mask=torch.ones_like(new_seg))
        
        response = self.tokenizer.decode(outputs[0]).split("<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]
        #response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def main():
    parser = argparse.ArgumentParser(description='Non Greedy Text Segmentation')
    parser.add_argument('--text', type=str, help='Input text to segment')
    parser.add_argument('--k', type=lambda x: int(x) if x.lower() != 'none' else None, default=10, help='Number of top segmentations to keep (use None for exhaustive search)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Perplexity normalization factor')
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help='Model name/path')
    parser.add_argument('--user_segmentation', type=str, help='User-defined segmentation (comma-separated tokens)')
    parser.add_argument('--generate_response', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True,
                         help='Whether to generate a response from the segmentation(s)')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmenter = TextSegmenter(k=args.k, alpha=args.alpha, model_name=args.model_name)

    if args.text:
        default_tokenization, default_perplexity = segmenter.get_default_tokenization_perplexity(args.text)
        print(f"Default Segmentation (Token List): {default_tokenization}")
        print(f"Default Perplexity: {default_perplexity}")

        if args.generate_response:
            # Generate response based on default segmentation
            response = segmenter.generate_response(default_tokenization)
            print(f"Generated Response for Default Tokenization: {response}")

        segs = segmenter.compute_segs(args.text)
        best_seg = segs[-1][0]
        print(f"Best Segmentation: {best_seg[0]}")
        tok_seg = segmenter.seg_to_token_list(best_seg[0])
        print(f"Best Segmentation (Token List): {tok_seg}")
        print(f"Best Perplexity: {best_seg[1]}")
        print(f"Runner Ups (Perplexity): {[seg[1] for seg in segs[-1][1:]]}")

        if args.generate_response:
            # Generate response based on best segmentation
            response = segmenter.generate_response(best_seg[0])
            print(f"Generated Response for Best Segmentation: {response}")
    
    if args.user_segmentation:
        user_seg = args.user_segmentation.split(",") # Todo: test with processing special tokens and processing without
        user_seg_perplexity = segmenter.calculate_seg_perplexity(user_seg)
        print(f"User Defined Segmentation: {user_seg}")
        print(f"User Defined Segmentation Perplexity: {user_seg_perplexity}")

        if args.generate_response:
            # Generate response based on user-defined segmentation
            response = segmenter.generate_response(user_seg)
            print(f"Generated Response for User Segmentation: {response}")

if __name__ == "__main__":
    main()