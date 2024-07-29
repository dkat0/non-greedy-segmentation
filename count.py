from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def remove_tag(question, tag):
    return question.replace(f"<{tag}>", "").replace(f"</{tag}>", "")

def extract_tag(question, tag):
    # tags are like <count></count>
    # this does not check for the validity of the tag
    # this does not work if there are multiple occurences of the tag
    if tag not in question:
        return None
    return question.split(f"<{tag}>")[1].split(f"</{tag}>")[0]

def llm_count(question, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):    
    messages = [
        {"role": "user", "content": question}
    ]
    
    char_to_be_counted = extract_tag(question, "char")
    question = remove_tag(question, "char")
    
    chat_template = tokenizer.apply_chat_template(messages, tokenize=False)
    
    left_str = chat_template.split("<count>")[0]
    right_str = chat_template.split("</count>")[1]
    
    count_word = extract_tag(question, "count")
    count_chars = list(count_word)
    if left_str[-1] == " ":
        left_str = left_str[:-1]
        count_chars[0] = " " + count_chars[0]
    
    left_tokens = tokenizer(left_str, return_tensors="pt").to("cuda")["input_ids"]
    right_tokens = tokenizer(right_str, return_tensors="pt").to("cuda")["input_ids"]
    
    char_level_seg = tokenizer(count_chars, is_split_into_words=True, return_tensors="pt").to("cuda")["input_ids"]
    
    start_token_id = model.config.bos_token_id # for Llama-3 models, this is 128000
    # Remove all start tokens, since the tokenizer will incorrectly add one to the beginning of all of these
    for _ in range(2): # Since the Llama tokenizer seems to incorrectly add two start tokens when using chat templating
        if left_tokens[0, 0] == start_token_id:
            left_tokens = left_tokens[:, 1:]
    if char_level_seg[0, 0] == start_token_id:
        char_level_seg = char_level_seg[:, 1:]
    if right_tokens[0, 0] == start_token_id:
        right_tokens = right_tokens[:, 1:]

    new_seg = torch.concat([torch.tensor([[start_token_id]]).to("cuda"), left_tokens, char_level_seg, right_tokens], dim=1)
    
    pure_question = left_str + ''.join(count_chars) + right_str
    original_seg = tokenizer(pure_question, return_tensors="pt").to("cuda")["input_ids"]
    
    original_response = model.generate(original_seg, attention_mask=torch.ones_like(original_seg))
    new_response = model.generate(new_seg, attention_mask=torch.ones_like(new_seg))
    
    original_response = tokenizer.decode(original_response[0]).split("<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]
    new_response = tokenizer.decode(new_response[0]).split("<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]
    
    original_answer = extract_tag(original_response, "answer")
    new_answer = extract_tag(new_response, "answer")
    gold_answer = sum([c == char_to_be_counted for c in count_word])
    
    return {
        "original_response": original_response,
        "new_response": new_response,
        "original_answer": original_answer,
        "new_answer": new_answer,
        "gold_answer": gold_answer,
        "original_correct": original_answer == gold_answer,
        "new_correct": new_answer == gold_answer
    }

if __name__ == "__main__":
    import pprint
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    
    while True:
        question = input(">> ").strip() 
        # example question: "put your answer in <answer></answer>: how many <char>r</char>s are there in <count>strawberry</count>"
        if question == "exit":
            break
        pprint.pprint(llm_count(question, tokenizer, model))