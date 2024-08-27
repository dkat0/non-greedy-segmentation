import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch.nn import CrossEntropyLoss

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List

class PerplexityCalculator:
    def __init__(self, model_name: str, normalization_alpha: float = 1.0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.perplexity_normalization_alpha = normalization_alpha
        
    def calculate_perplexities(self, sentences: List[str]) -> List[float]:
        """
        Calculates perplexity scores for a list of sentences.
        """
        # Tokenize input sentences
        tensor_input = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to("cuda")
        
        input_ids = tensor_input['input_ids']
        attention_mask = tensor_input['attention_mask']
        
        # Get model logits
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask) # Need padding=True or not?
            logits = outputs.logits
        
        # PPL calculation modified from:
        # https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/models/llama/modeling_llama.py#L1095

        # Shift logits and labels for computing loss:

        # Removing the last token's logits -- the model predicts the next token, and the last token has no next token to predict.
        shift_logits = logits[..., :-1, :].contiguous()
        # Removing the first token because there is no token before the first one to predict.
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Flatten the tensors
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        
        # Compute the loss
        # Setting reduction='none' ensures that the loss is computed for each token individually, rather than averaging the losses.
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits, shift_labels)
        
        # Reshape loss to match input_ids shape, but with one less token due to the shifting step.
        loss = loss.view(input_ids.size(0), input_ids.size(1) - 1)
        
        # Mask the loss to only consider the actual words (ignore padding)
        loss = loss * attention_mask[:, 1:].contiguous()
        
        # Calculate perplexity for each sentence
        perplexities = []
        for sentence_loss in loss:
            sentence_loss = sentence_loss[sentence_loss != 0]  # Remove padding losses
            num_tokens = sentence_loss.size(0)
            average_loss = torch.sum(sentence_loss) / num_tokens # (num_tokens ** self.perplexity_normalization_alpha)
            normalized_ppl = torch.exp(average_loss)
            perplexities.append(normalized_ppl.item())
        
        return perplexities
    
    """
    Calculates a perplexity score for a sentence.
    """
    def calculate_perplexity(self, sentence: str) -> float:
        tensor_input = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to("cuda")
        input_ids = tensor_input['input_ids']
        attention_mask = tensor_input['attention_mask']
        
        # Calculate Loss
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
            loss = outputs.loss

        # Calculate Normalized PPL
        num_tokens = input_ids.shape[1]
        average_loss = loss # / num_tokens # / (num_tokens ** self.perplexity_normalization_alpha)
        normalized_ppl = torch.exp(average_loss)
        
        return normalized_ppl.item() 

model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
calculator = PerplexityCalculator(model_name, normalization_alpha=0.5)
sentences = ["Hello world", "Hello joe", "Hello potato"]
perplexities = calculator.calculate_perplexities(sentences)
print(perplexities)
# [24118.5703125, 12406401.0, 6520770.5]

# Are these the same?
for sentence in sentences:
    print(calculator.calculate_perplexity(sentence))
#61.530181884765625
#786.9901733398438
#605.2378540039062

# different numbers, same ordering?