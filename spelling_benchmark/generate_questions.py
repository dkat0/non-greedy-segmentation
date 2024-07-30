import json
import random
import inflect
from gibberish import Gibberish
from collections import Counter

# Config
word_selection_char_repetition_propensity = 2
letter_selection_char_repetition_propensity = 3
num_questions_to_generate = 10000

# Load word lists
def load_words(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file.readlines()]

words_short = load_words('word_list/google-10000-english-usa-no-swears-short.txt')
words_medium = load_words('word_list/google-10000-english-usa-no-swears-medium.txt')
words_long = load_words('word_list/google-10000-english-usa-no-swears-long.txt')

words_short = [word for word in words_short if len(word) > 1] # remove individual characters

# Load base questions
with open('question_templates.json', 'r') as file:
    question_templates = json.load(file)

p = inflect.engine()
gib = Gibberish()

def ordinal(n):
    return p.ordinal(n)

def generate_question(template, target_word, letter=None, index1=None, index2=None, replacement_value=None, n=None, substring=None, capitalized_letter=None):
    question = template
    if letter:
        question = question.replace('<letter>', letter)
    if target_word:
        question = question.replace('<target_word>', target_word)
    if index1 is not None:
        question = question.replace('<index1>', str(index1))
        question = question.replace('<index1_with_ordinal_suffix>', ordinal(index1))
    if index2 is not None:
        question = question.replace('<index2>', str(index2))
        question = question.replace('<index2_with_ordinal_suffix>', ordinal(index2))
    if replacement_value:
        question = question.replace('<replacement_value>', replacement_value)
    if n is not None:
        question = question.replace('<n>', str(n))
        question = question.replace('<n_with_ordinal_suffix>', ordinal(n))
    if substring is not None:
        question = question.replace('<substring>', substring)
    if capitalized_letter is not None:
        question = question.replace('<capitalized_letter>', capitalized_letter)
    return question

# Util functions

def generate_random_substring(target_word, word_list):
    while True:
        random_word = random.choice(word_list)
        if random_word not in target_word:
            start_index = random.randint(0, len(random_word) - 2)
            end_index = random.randint(start_index + 1, len(random_word))
            substring = random_word[start_index:end_index]
            if substring not in target_word:
                return substring

def max_char_repetition(word):
    counter = Counter(word)
    return max(counter.values())

def get_weighted_letter_choice(word, weight_power=letter_selection_char_repetition_propensity):
    letter_counts = Counter(word)
    letters = list(letter_counts.keys())
    weights = [count ** weight_power for count in letter_counts.values()]
    return random.choices(letters, weights=weights, k=1)[0]

def count_letter_in_word(word, letter):
    return word.count(letter)

def location_of_letter_in_word(word, letter):
    return [i+1 for i, l in enumerate(word) if l == letter]

def letter_after_nth_occurrence(word, letter, n):
    index = n - 1
    positions = [i for i, l in enumerate(word) if l == letter]
    if len(positions) > index:
        return word[positions[index] + 1] if positions[index] + 1 < len(word) else None
    return None

def letters_not_in_word(word):
    return [chr(i) for i in range(97, 123) if chr(i) not in word]

def spell_word_backwards(word):
    return word[::-1]

def extract_substring(word, index1, index2):
    return word[index1:index2+1]

def replace_letter_in_word(word, letter, replacement_value):
    return word.replace(letter, replacement_value)

def capitalize_specific_letters(word, letter):
    return word.replace(letter, letter.upper())

def substring_existence(word, substring):
    return substring in word

def count_total_letters(word):
    return len(word)

question_types = question_templates['question_types']
all_words = words_short + words_medium + words_long
weighted_words = [(word, max_char_repetition(word) ** word_selection_char_repetition_propensity) for word in all_words]

# Extract words and their respective weights
words, weights = zip(*weighted_words)

generated_questions = []

# Set number of questions to generate
for id in range(1, num_questions_to_generate+1):
    q_type = random.choice(list(question_types.keys()))
    template = random.choice(question_types[q_type])

    if random.random() < 0.8:
        #target_word = random.choice(all_words).lower()
        target_word = random.choices(words, weights=weights, k=1)[0].lower()

        if target_word in words_short:
            word_length_category = "short"
        elif target_word in words_medium:
            word_length_category = "medium"
        else:
            word_length_category = "long"

        is_pseudoword = False
    else:
        if random.random() < 0.5:
            word_length_category = "medium"
            consonant_repeats = 1
        else:
            word_length_category = "long"
            consonant_repeats = 2

        target_word = gib.generate_word(vowel_consonant_repeats=consonant_repeats)
        is_pseudoword = True
    
    if q_type == 'count_letter_in_word':
        letter = get_weighted_letter_choice(target_word)
        answer = str(count_letter_in_word(target_word, letter))
        question = generate_question(template, target_word, letter=letter)
    elif q_type == 'location_of_letter_in_word':
        letter = get_weighted_letter_choice(target_word)
        answer = ", ".join([str(loc) for loc in location_of_letter_in_word(target_word, letter)])
        question = generate_question(template, target_word, letter=letter)
    elif q_type == 'letter_after_nth_occurrence':
        while 1:
            letter = get_weighted_letter_choice(target_word)
            n = random.randint(1, target_word.count(letter))
            answer = letter_after_nth_occurrence(target_word, letter, n)
            if answer is not None:
                break # if answer doesn't exist (i.e. letter was the last letter of the word), try again
        question = generate_question(template, target_word, letter=letter, n=n)
    elif q_type == 'letters_not_in_word':
        answer = ", ".join(letters_not_in_word(target_word))
        question = generate_question(template, target_word)
    elif q_type == 'spell_word_backwards':
        answer = spell_word_backwards(target_word)
        question = generate_question(template, target_word)
    elif q_type == 'extract_substring':
        index1 = random.randint(0, len(target_word) - 2)
        index2 = random.randint(index1 + 1, len(target_word) - 1)
        answer = extract_substring(target_word, index1, index2)
        question = generate_question(template, target_word, index1=index1+1, index2=index2+1)
    elif q_type == 'replace_letter_in_word':
        letter = get_weighted_letter_choice(target_word)
        replacement_value = random.choice('abcdefghijklmnopqrstuvwxyz')
        answer = replace_letter_in_word(target_word, letter, replacement_value)
        question = generate_question(template, target_word, letter=letter, replacement_value=replacement_value)
    elif q_type == 'capitalize_specific_letters':
        letter = get_weighted_letter_choice(target_word)
        answer = capitalize_specific_letters(target_word, letter)
        question = generate_question(template, target_word, letter=letter, capitalized_letter=letter.upper())
    elif q_type == 'substring_existence':
        if random.random() < 0.5:
            # 50% chance to generate a substring that is not in the target word
            substring = generate_random_substring(target_word, words_medium)
            answer = "No"
        else:
            # 50% chance to gen a substring that is in the word
            start_index = random.randint(0, len(target_word) - 1)
            end_index = random.randint(start_index + 1, len(target_word))
            substring = target_word[start_index:end_index]
            answer = "Yes"
        
        question = generate_question(template, target_word, substring=substring)
    elif q_type == 'count_total_letters':
        answer = str(count_total_letters(target_word))
        question = generate_question(template, target_word)
    
    generated_questions.append({
        "id": id, 
        "question": question,
        "target_word": target_word,
        "answer": answer,
        "question_type": q_type,
        "word_length_category": word_length_category,
        "is_pseudoword": is_pseudoword
    })

# Save generated questions to JSON file
with open('questions/generated_questions.json', 'w') as file:
    json.dump(generated_questions, file, indent=2)
