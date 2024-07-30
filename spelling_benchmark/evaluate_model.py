from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pprint
import os 
from json_encoder import CompactJSONEncoder
from datetime import datetime

class ModelEvaluator:
    def __init__(self, model_name, questions_file, tokenization_modes, segmenter_params):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id # to suppress the warning

        self.questions = self.load_questions(questions_file)
        self.tokenization_modes = tokenization_modes
        self.segmenter_params = segmenter_params

        self.segmenter = None
        if 'new_whole_prompt' in tokenization_modes or 'new_target_word' in tokenization_modes:
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

    def prepare_tokens(self, model_prompt, chat_template, question, tokenization_mode):
        start_token_id = self.model.config.bos_token_id # for Llama-3 models, this is 128000

        if tokenization_mode == "default":
            tokens = self.tokenizer(chat_template, return_tensors="pt").to(self.device)["input_ids"]

            # This part is simply for ease of accessing the tokenization for the target word for more detailed results, not really necessary at all
            # This might not be completely accurate due to potentially prepending spaces, will all tokenizers do that? Not sure, might be a better way for this
            try:
                potential_space_index = model_prompt.index(question['target_word']) - 1
                word_to_tokenize = " " + question['target_word'] if model_prompt[potential_space_index] == " " else question['target_word']
                word_tokens = self.tokenizer(word_to_tokenize, return_tensors="pt").to(self.device)["input_ids"]

                if word_tokens[0, 0] == start_token_id:
                    word_tokens = word_tokens[:, 1:]
                
                return tokens, word_tokens
            except ValueError:
                return tokens, None
        
        if tokenization_mode in ["new_whole_prompt", "new_target_word", "characterize_target_word"]:
            if tokenization_mode == "new_whole_prompt":
                seg = self.segmenter.tokenize(model_prompt)
                left_text, right_text = chat_template.split(model_prompt)
            else:
                seg = self.segmenter.tokenize(question['target_word']) if tokenization_mode == "new_target_word" else list(question['target_word'])
                left_text, right_text = chat_template.split(question['target_word'], 1)

                # Move previous space into seg
                if left_text[-1] == " ":
                    left_text = left_text[:-1]
                    seg[0] = " " + seg[0]

            left_tokens = self.tokenizer(left_text, return_tensors="pt").to(self.device)["input_ids"]
            text_tokens = self.tokenizer(seg, is_split_into_words=True, return_tensors="pt").to(self.device)["input_ids"]
            right_tokens = self.tokenizer(right_text, return_tensors="pt").to(self.device)["input_ids"]
            
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
        results = {mode: [] for mode in self.tokenization_modes}
        correct_answers = {mode: 0 for mode in self.tokenization_modes}
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        os.mkdir(f"results/evaluation_results_{timestamp}")
        output_files = {mode: f"results/evaluation_results_{timestamp}/{mode}.json" for mode in self.tokenization_modes}

        # Create all files
        for file in output_files.values():
            with open(file, "x") as _:
                pass
        
        print("[info] Starting evaluation. See /results/ folder for more detailed information.\n")
        for i, question in enumerate(self.questions):
            model_prompt = "Provide your answer inside <answer></answer> tags: " + question['question']
            print(f"[{question['id']}] Prompt {question['id']}: {model_prompt}")

            for tokenization_mode in self.tokenization_modes:
                messages = [{"role": "user", "content": model_prompt}]
                chat_template = self.tokenizer.apply_chat_template(messages, tokenize=False)
                #print("Chat template: ", chat_template)
                
                tokens, text_tokens = self.prepare_tokens(model_prompt, chat_template, question, tokenization_mode)
                #print("Got tokens: ", tokens)

                with torch.no_grad():
                    outputs = self.model.generate(tokens, attention_mask=torch.ones_like(tokens), max_new_tokens=50)
                
                model_answer = self.tokenizer.decode(outputs[0]).split("<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]
                #response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                evaluation_result = {
                    "id": question["id"],
                    "question": question["question"],
                    "expected_answer": question["answer"],
                    "model_parsed_answer": self.parse_answer(model_answer),
                    "model_response": model_answer,
                    "correct": self.evaluate_answer(question, model_answer)
                }
                if evaluation_result["correct"]:
                    correct_answers[tokenization_mode] += 1
                accuracy = correct_answers[tokenization_mode] / (i + 1)
                evaluation_result["current_total_accuracy"] = accuracy

                #pprint.pprint(evaluation_result)

                token_list = self.tokenizer.convert_ids_to_tokens(tokens.tolist()[0])
                tokenization_info = {
                    "tokenization_mode": tokenization_mode,
                    "tokens": [self.decode_text_special(token) for token in token_list]
                }
                if tokenization_mode in ["default", "new_target_word", "characterize_target_word"]:
                    word_tokens = self.tokenizer.convert_ids_to_tokens(text_tokens.tolist()[0])
                    tokenization_info["target_word_tokens"] = [self.decode_text_special(token) for token in word_tokens]
                
                if tokenization_mode in ["new_whole_prompt", "new_target_word"]:
                    tokenization_info["segmenter_params"] = self.segmenter_params
                
                evaluation_result["tokenization_info"] = tokenization_info

                results[tokenization_mode].append(evaluation_result)

                print(f"[{question['id']}] {tokenization_mode} mode: {'correct' if evaluation_result['correct'] else 'incorrect'}")

                with open(output_files[tokenization_mode], 'w') as f:
                    json.dump(results[tokenization_mode], f, cls=CompactJSONEncoder)
            
            print("")
        
        return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Model Evaluator')
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help='Model name/path')
    parser.add_argument("--questions", type=str, default="questions/generated_questions.json", help='Path to questions JSON file')
    parser.add_argument('--tokenization_modes', type=str, nargs='+', choices=['default', 'new_whole_prompt', 'new_target_word', 'characterize_target_word'], default=['default'])
    parser.add_argument('--k', type=lambda x: int(x) if x.lower() != 'none' else None, default=10, help='Number of top segmentations to keep (use None for exhaustive search)')
    parser.add_argument('--alpha', type=float, default=0.4, help='Perplexity normalization factor (if using new tokenization mode)')
    args = parser.parse_args()

    segmenter_params = {
        "k": args.k,
        "alpha": args.alpha
    }
    
    evaluator = ModelEvaluator(args.model_name, args.questions, args.tokenization_modes, segmenter_params)
    results = evaluator.evaluate_model()
