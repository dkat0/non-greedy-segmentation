# non-greedy-segmentation
Improving the performance of LLMs on math, coding, and spelling benchmarks by modifying the tokenization process at inference time.


# CLI example commands

**Compare Default vs. Perplexity-Optimized Segmentation:**
```
python tokenizer.py --text "How many rs are there in strawberry" --k 10 --alpha 0.5 --model_name "meta-llama/Meta-Llama-3-8B"
```

**Obtain Perplexity of User-Defined Segmentation (comma-seperated)**
```
python tokenizer.py --user_segmentation "How, many, rs, are, there, in, straw,berry" --alpha 0.5 --model_name "meta-llama/Meta-Llama-3-8B"
```
This tests the perplexity of the segmentation `["How", " many", " rs", " are", " there", " in", " straw", "berry"]`


# EvalPlus test:
Install the nightly version of evalplus

```
pip install "git+https://github.com/evalplus/evalplus.git" --upgrade
cd evalplus
pip install -r codegen/requirements.txt
python codegen/generate.py --model "meta-llama/Meta-Llama-3-8B" --greedy --root res --dataset humaneval --backend hf --new_tokenization True
```
