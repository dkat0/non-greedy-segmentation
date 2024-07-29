from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pprint
from json_encoder import CompactJSONEncoder
from datetime import datetime

class ModelEvaluator:
    def __init__(self, model_name, questions_file, tokenization_mode, segmenter_params):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.questions = self.load_questions(questions_file)
        self.tokenization_mode = tokenization_mode

        self.segmenter_params = segmenter_params

        self.segmenter = None
        if tokenization_mode in ['new_whole_prompt', 'new_target_word']:
            self.segmenter = self.initialize_segmenter()

    @staticmethod
    def load_questions(questions_file):
        with open(questions_file, 'r') as file:
            return json.load(file)
    
    def initialize_segmenter(self):
        import os
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)

        sys.path.append(parent_dir)
        from tokenizer import TextSegmenter
        sys.path.remove(parent_dir)

        return TextSegmenter(k=self.segmenter_params["k"], alpha=self.segmenter_params["alpha"], model=self.model, tokenizer=self.tokenizer)

    @staticmethod
    def remove_tag(question, tag):
        return question.replace(f"<{tag}>", "").replace(f"</{tag}>", "")

    @staticmethod
    def extract_tag(question, tag):
        if f"<{tag}>" not in question:
            return None
        return question.split(f"<{tag}>")[1].split(f"</{tag}>")[0]

    @staticmethod
    def decode_text_special(text):
        return text.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')

    @staticmethod
    def remove_first_occurrence(main_string, substring):
        index = main_string.find(substring)
        if index != -1:
            return main_string[:index] + main_string[index + len(substring):]
        return main_string

    def parse_answer(self, model_answer):
        answer = self.extract_tag(model_answer, "answer")
        if answer is None:
            return None
        return answer.strip().replace(", ", ",").replace(",", ", ")

    def evaluate_answer(self, question, model_answer):
        return self.parse_answer(model_answer).replace(", ", ",") == question['answer'].replace(", ", ",")

    def prepare_tokens(self, model_prompt, chat_template, question):
        if self.tokenization_mode == "default":
            tokens = self.tokenizer(model_prompt, return_tensors="pt").to(self.device)["input_ids"]
            return tokens, None
        
        if self.tokenization_mode in ["new_whole_prompt", "new_target_word", "characterize_target_word"]:
            if self.tokenization_mode == "new_whole_prompt":
                seg = self.segmenter.tokenize(model_prompt)
                left_text, right_text = chat_template.split(model_prompt)
            else:
                seg = self.segmenter.tokenize(question['target_word']) if self.tokenization_mode == "new_target_word" else list(question['target_word'])
                left_text, right_text = chat_template.split(question['target_word'], 1)

                # Move previous space into seg
                if left_text[-1] == " ":
                    left_text = left_text[:-1]
                    seg[0] = " " + seg[0]

            left_tokens = self.tokenizer(left_text, return_tensors="pt").to(self.device)["input_ids"]
            text_tokens = self.tokenizer(seg, is_split_into_words=True, return_tensors="pt").to(self.device)["input_ids"]
            right_tokens = self.tokenizer(right_text, return_tensors="pt").to(self.device)["input_ids"]
            
            start_token_id = self.model.config.bos_token_id # for Llama-3 models, this is 128000

            # Remove all start tokens, since the tokenizer will incorrectly add one to the beginning of all of these
            for _ in range(2): # Since the Llama tokenizer seems to incorrectly add two start tokens when using chat templating
                if left_tokens[0, 0] == start_token_id:
                    left_tokens = left_tokens[:, 1:]
            if text_tokens[0, 0] == start_token_id:
                text_tokens = text_tokens[:, 1:]
            if right_tokens[0, 0] == start_token_id:
                right_tokens = right_tokens[:, 1:]
            
            # Concatenate tokens, adding the start token only once at the beginning
            tokens = torch.concat([torch.tensor([[start_token_id]]).to(self.device), left_tokens, text_tokens, right_tokens], dim=1)
            return tokens, text_tokens

    def evaluate_model(self):
        results = []
        correct_answers = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_results_{timestamp}.json"
        
        for i, question in enumerate(self.questions):
            model_prompt = "Provide your answer inside <answer></answer> tags: " + question['question']
            print(f"Querying model for Prompt {question['id']}: {model_prompt}")

            messages = [{"role": "user", "content": model_prompt}]
            chat_template = self.tokenizer.apply_chat_template(messages, tokenize=False)
            print("Chat template: ", chat_template)
            
            tokens, text_tokens = self.prepare_tokens(model_prompt, chat_template, question)
            print("Got tokens: ", tokens)

            with torch.no_grad():
                outputs = self.model.generate(tokens, attention_mask=torch.ones_like(tokens), max_new_tokens=50)
            
            model_answer = self.tokenizer.decode(outputs[0]).split("<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]
            #response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            evaluation_result = {
                "id": question["id"],
                "question": question["question"],
                "expected_answer": question["answer"],
                "model_parsed_answer": self.parse_answer(model_answer).replace(",", ", "),
                "model_response": model_answer,
                "correct": self.evaluate_answer(question, model_answer)
            }
            if evaluation_result["correct"]:
                correct_answers += 1
            accuracy = correct_answers / (i + 1)
            evaluation_result["current_total_accuracy"] = accuracy

            pprint.pprint(evaluation_result)

            token_list = self.tokenizer.convert_ids_to_tokens(tokens.tolist()[0])
            tokenization_info = {
                "tokenization_mode": self.tokenization_mode,
                "tokens": [self.decode_text_special(token) for token in token_list]
            }
            if self.tokenization_mode in ["new_target_word", "characterize_target_word"]:
                word_tokens = self.tokenizer.convert_ids_to_tokens(text_tokens.tolist()[0])
                tokenization_info["target_word_tokens"] = [self.decode_text_special(token) for token in word_tokens]
            
            if self.tokenization_mode in ["new_whole_prompt", "new_target_word"]:
                tokenization_info["segmenter_params"] = self.segmenter_params
            
            evaluation_result["tokenization_info"] = tokenization_info

            results.append(evaluation_result)

            with open(output_file, 'w') as f:
                json.dump(results, f, cls=CompactJSONEncoder)
        
        return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Model Evaluator')
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help='Model name/path')
    parser.add_argument("--questions", type=str, default="generated_questions.json", help='Path to questions JSON file')
    parser.add_argument('--tokenization_mode', type=str, choices=['default', 'new_whole_prompt', 'new_target_word', 'characterize_target_word'], default='default')
    parser.add_argument('--k', type=lambda x: int(x) if x.lower() != 'none' else None, default=10, help='Number of top segmentations to keep (use None for exhaustive search)')
    parser.add_argument('--alpha', type=float, default=0.4, help='Perplexity normalization factor (if using new tokenization mode)')
    args = parser.parse_args()

    segmenter_params = {
        "k": args.k,
        "alpha": args.alpha
    }
    
    evaluator = ModelEvaluator(args.model_name, args.questions, args.tokenization_mode, segmenter_params)
    results = evaluator.evaluate_model()
