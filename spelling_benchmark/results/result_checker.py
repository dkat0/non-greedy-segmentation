import json
from json_encoder import CompactJSONEncoder
from collections import defaultdict


# Define the folder and files
folder_path = 'evaluation_results_2024-07-30_013912'
files = ['characterize_target_word.json', 'default.json', 'new_target_word.json']
questions_file = 'generated_questions.json'

# Hold counts of correct answers by question type and tokenization mode
correct_counts = defaultdict(lambda: defaultdict(int))
correct_mode_count = {}
total_count = 0

# Load the data from the files
data = {}
for file in files:
    with open(f"{folder_path}/{file}") as f:
        mode = file.split('.')[0]
        data[mode] = json.load(f)
    correct_mode_count[mode] = 0
    

with open(questions_file) as f:
    questions_data  = json.load(f)

question_type_map = {q['id']: q['question_type'] for q in questions_data}

# Compare the correctness of each entry across the different modes
results = []
for entry in data['characterize_target_word']:
    entry_id = entry['id']
    question_type = question_type_map[entry_id]
    entry_result = {
        "id": entry_id,
        "question": entry['question'],
        "question_type": question_type,
        "expected_answer": entry['expected_answer'],
        "modes": {}
    }
    all_correct = True
    all_incorrect = True
    for mode in data:
        mode_entry = next(e for e in data[mode] if e['id'] == entry_id)

        entry_result['modes'][mode] = {
            "correct": mode_entry['correct'],
            "model_response": mode_entry['model_response'],
            "tokens": mode_entry['tokenization_info']['tokens'],
            "target_word_tokens": mode_entry['tokenization_info']['target_word_tokens']
        }

        # Update the correct counts
        if mode_entry['correct']:
            correct_counts[question_type][mode] += 1
        
        if mode_entry['correct']:
            correct_mode_count[mode] += 1

        if mode_entry['correct']:
            all_incorrect = False
        else:
            all_correct = False
    
    total_count += 1
    
    if not (all_correct or all_incorrect):
        results.append(entry_result)

# Save the results
output_file = f"{folder_path}/inconsistent_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4, cls=CompactJSONEncoder)

print(f'Results saved to {output_file}')

# Print the correct counts by question type and tokenization mode
print("Correct counts by question type and tokenization mode:")
for question_type, mode_counts in correct_counts.items():
    print(f"Question type: {question_type}")
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count}")

print("\nTotal accuracies for each mode:")
for mode, correct_amount in correct_mode_count.items():
    accuracy = correct_amount / total_count
    print(f"{mode}: {accuracy}%, correct_amount = {correct_amount}/{total_count}")
