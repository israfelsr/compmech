# QwenVLLMWrapper Implementation Walkthrough

This document explains the `QwenVLLMWrapper` implementation step by step to help you understand how it works and how to adapt it for other models.

## Overview

The wrapper integrates Qwen2-VL with vLLM for fast inference in the AdaptVis evaluation framework. It follows the same interface as the LLaVA wrapper so it can be used as a drop-in replacement.

---

## Part 1: Initialization (`__init__`)

### Purpose
Set up the vLLM engine and configure it for Qwen2-VL.

### Code Breakdown

```python
def __init__(
    self,
    root_dir,
    device,
    method="base",
    model_name="/leonardo_work/EUHPC_D27_102/compmech/models/Qwen2.5-3B",
):
```

**Parameters:**
- `root_dir`: Cache directory (not used by vLLM but kept for API compatibility)
- `device`: GPU device (vLLM handles device placement automatically)
- `method`: Evaluation method (kept for compatibility with LLaVA's adaptive methods)
- `model_name`: Path to Qwen model checkpoint

```python
# Get vLLM-specific model configuration
vllm_config = self._get_vllm_config(model_name)
```

This calls a helper method to get optimal vLLM settings for Qwen2-VL.

```python
# Initialize vLLM model
self.llm = LLM(model=model_name, **vllm_config)
```

**Key Point:** This creates the vLLM engine. vLLM automatically:
- Loads the model to GPU
- Sets up PagedAttention for efficient memory
- Prepares batching infrastructure

```python
# Set sampling parameters for generation
self.sampling_params = SamplingParams(
    temperature=0.0,  # Greedy decoding for consistent results
    max_tokens=100,
    stop_token_ids=None,
)
```

**Sampling Parameters:**
- `temperature=0.0`: Greedy decoding (always pick most likely token)
- `max_tokens=100`: Maximum length of generated answer
- `stop_token_ids=None`: No special stopping tokens

---

## Part 2: vLLM Configuration (`_get_vllm_config`)

### Purpose
Configure vLLM for optimal Qwen2-VL performance.

### Code

```python
def _get_vllm_config(self, model_name):
    config = {
        "max_model_len": 4096,      # Maximum sequence length
        "max_num_seqs": 5,           # Batch size (how many requests to process together)
        "limit_mm_per_prompt": {"image": 1},  # One image per prompt
    }

    config["mm_processor_kwargs"] = {
        "min_pixels": 28 * 28,           # Minimum image resolution
        "max_pixels": 1280 * 28 * 28,    # Maximum image resolution
    }

    return config
```

**Key Parameters:**
- `max_model_len`: Qwen2-VL context window (4096 tokens)
- `max_num_seqs`: How many prompts vLLM processes in parallel (higher = more GPU memory but faster)
- `mm_processor_kwargs`: Tells vLLM what image sizes to expect

**Why these values?**
- Qwen2-VL uses dynamic resolution from 28x28 to 1280x28x28 pixels
- `max_num_seqs=5` is a good balance between speed and memory for a 7B model

---

## Part 3: Prompt Formatting (`_format_qwen_prompt`)

### Purpose
Convert a question into Qwen2-VL's chat template format.

### Code

```python
def _format_qwen_prompt(self, question):
    prompt_text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt_text
```

**Template Structure:**
1. **System message**: `<|im_start|>system\n...<|im_end|>`
   - Sets the assistant's role
2. **User message**: `<|im_start|>user\n...<|im_end|>`
   - Contains the vision tokens and question
3. **Vision tokens**: `<|vision_start|><|image_pad|><|vision_end|>`
   - Tells the model where the image is
4. **Assistant prompt**: `<|im_start|>assistant\n`
   - Prompts the model to start answering

**Example:**
Input: `"What is on the left of the apple?"`

Output:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>What is on the left of the apple?<|im_end|>
<|im_start|>assistant
```

---

## Part 4: Creating vLLM Prompts (`_create_vllm_prompt`)

### Purpose
Package the formatted prompt and image into vLLM's expected format.

### Code

```python
def _create_vllm_prompt(self, prompt, image):
    formatted_prompt = self._format_qwen_prompt(prompt)

    vllm_prompt = {
        "prompt": formatted_prompt,
        "multi_modal_data": {"image": image.convert("RGB")},
    }
    return vllm_prompt
```

**Output Format:**
```python
{
    "prompt": "<|im_start|>system\n...",  # Formatted text
    "multi_modal_data": {
        "image": PIL.Image  # PIL Image object in RGB
    }
}
```

**Why this format?**
- vLLM expects a dictionary with `prompt` (text) and `multi_modal_data` (images)
- The image must be a PIL Image in RGB format
- vLLM handles the actual image encoding internally

---

## Part 5: Main Evaluation Loop (`get_out_scores_wh_batched`)

This is the core method that runs evaluation. Let me break it into sections:

### Section 5.1: Setup

```python
scores = []
index_of_total = 0
acc = 0
correct_id = []
```

**Tracking variables:**
- `scores`: Stores prediction scores for each sample
- `index_of_total`: Current sample number
- `acc`: Running accuracy count
- `correct_id`: List of correctly predicted sample indices

### Section 5.2: Load Questions and Answers

```python
qst_ans_file = f"prompts/{dataset}_with_answer_{option}_options.jsonl"

with open(qst_ans_file, "r") as file:
    prompt_list = []
    answer_list = []
    for line in file:
        data = json.loads(line)
        prompt_list.append(data["question"])
        answer_list.append(data["answer"])
```

**What's happening:**
1. Load a JSONL file containing questions and answers
2. Each line is like: `{"question": "What is left of X?", "answer": ["left"]}`
3. Build two lists: questions and corresponding answers

**Example JSONL:**
```json
{"question": "What is to the left of the apple?", "answer": ["banana"]}
{"question": "What is above the cat?", "answer": ["ceiling"]}
```

### Section 5.3: Sampling (Train/Test Split)

```python
if SAMPLE:
    idx_file_path = f"./output/sampled_idx_{dataset}.npy"

    if os.path.exists(idx_file_path):
        sampled_indices = np.load(idx_file_path).tolist()
    else:
        sampled_indices = random.sample(
            range(total_data_count), int(0.2 * total_data_count)
        )
        sampled_indices.sort()
        np.save(idx_file_path, np.array(sampled_indices))
```

**Purpose:** Split dataset into train/test (80/20 split)

**Logic:**
1. Check if we already sampled before (load from .npy file)
2. If not, randomly sample 20% of indices
3. Save indices for reproducibility
4. Use these same indices across all runs

**Why?** This ensures consistent evaluation across different models.

### Section 5.4: Batch Processing Loop

```python
for batch in tqdm(joint_loader):
    batch_scores = []

    # Iterate over each image option in the batch
    for i_option in batch["image_options"]:
        im_scores = []
```

**Structure:**
- `batch`: Contains multiple images from the dataset
- `batch["image_options"]`: List of images for this batch
- `i_option`: Each image in the current batch

**Example batch structure:**
```python
batch = {
    "image_options": [[img1, img2, img3, img4]],  # 4 images
    "caption_options": [...]
}
```

### Section 5.5: Prepare vLLM Batch

```python
# Collect all images and prompts for this batch to process together
vllm_prompts = []
for image in i_option:
    prompt = prompt_list[index_of_total]
    vllm_prompt = self._create_vllm_prompt(prompt, image)
    vllm_prompts.append(vllm_prompt)
    index_of_total += 1
```

**Key Insight:** This is where batching happens!
- Instead of processing one image at a time
- We collect all images in the current option group
- vLLM will process them all together in parallel

**Example:**
```python
vllm_prompts = [
    {"prompt": "...", "multi_modal_data": {"image": img1}},
    {"prompt": "...", "multi_modal_data": {"image": img2}},
    {"prompt": "...", "multi_modal_data": {"image": img3}},
    {"prompt": "...", "multi_modal_data": {"image": img4}},
]
```

### Section 5.6: Generate with vLLM

```python
outputs = self.llm.generate(vllm_prompts, self.sampling_params)
```

**This is the magic line!**
- vLLM takes all prompts at once
- Processes them in parallel using PagedAttention
- Returns all outputs together

**Output format:**
```python
outputs = [
    RequestOutput(outputs=[CompletionOutput(text="banana")]),
    RequestOutput(outputs=[CompletionOutput(text="ceiling")]),
    ...
]
```

### Section 5.7: Process Outputs

```python
local_idx = 0
for output in outputs:
    gen = output.outputs[0].text.strip()

    # Get corresponding prompt and answer
    batch_idx = index_of_total - len(i_option) + local_idx
    prompt = prompt_list[batch_idx]
    golden_answer = answer_list[batch_idx][0]
```

**What's happening:**
1. Extract generated text from vLLM output
2. Calculate which sample this corresponds to
3. Get the original prompt and correct answer

**Index calculation:**
- `index_of_total`: Current position after processing all images
- `len(i_option)`: How many images we just processed
- `local_idx`: Which output we're currently looking at
- Result: `index_of_total - len(i_option) + local_idx` gives the correct index

### Section 5.8: Check Answer and Score

```python
if len(list(c_option)) == 4:
    if (
        golden_answer in gen
        or golden_answer.lower() in gen.lower()
    ) and not (
        golden_answer.lower() == "on"
        and "front" in gen.strip().lower()
    ):
        acc += 1
        correct_id.append(batch_idx)
        answers = [1, 0, 0, 0]
    else:
        answers = [0, 0, 1, 0]
```

**Scoring logic:**
1. Check if golden answer appears in generated text
2. Special case: "on" shouldn't match "front" (to avoid false positives)
3. If correct: `answers = [1, 0, 0, 0]` (first option is correct)
4. If wrong: `answers = [0, 0, 1, 0]` (mark as incorrect)

**Why this format?**
- Matches the original AdaptVis scoring system
- First position = correct, others = incorrect options

### Section 5.9: Save Results

```python
# Save results periodically
output_file_path = f"./output/results_qwen_vllm_{dataset}_{method}_{option}option_{TEST}.json"
with open(output_file_path, "w", encoding="utf-8") as fout:
    json.dump(results, fout, ensure_ascii=False, indent=4)
```

**Purpose:** Save after each batch so progress isn't lost if the job crashes.

**Output JSON:**
```json
[
    {
        "Prompt": "What is to the left of the apple?",
        "Generation": "banana",
        "Golden": "banana"
    },
    ...
]
```

---

## Key Differences: vLLM vs Standard Transformers

| Aspect | Standard Transformers | vLLM |
|--------|----------------------|------|
| **Batching** | Manual, requires careful memory management | Automatic, optimized |
| **Memory** | Loads full KV cache for each sequence | PagedAttention (shared memory) |
| **Speed** | Slower, sequential processing | 10-20x faster with batching |
| **API** | `model.generate()` per sample | `llm.generate()` for batch |
| **Setup** | Simple `from_pretrained()` | Requires config dict |

---

## How to Adapt This for Other Models

### Step 1: Update `__init__`
```python
def __init__(self, ...):
    # Change model path
    self.llm = LLM(model="your-model-path", **vllm_config)
```

### Step 2: Update `_format_prompt`
```python
def _format_your_model_prompt(self, question):
    # Use your model's chat template
    prompt_text = f"<your_model_format>{question}</your_model_format>"
    return prompt_text
```

### Step 3: Update `_get_vllm_config`
```python
def _get_vllm_config(self, model_name):
    config = {
        "max_model_len": YOUR_CONTEXT_LENGTH,
        "max_num_seqs": YOUR_BATCH_SIZE,
        # Add model-specific configs
    }
    return config
```

### Step 4: Test!
```python
python AdaptVis/main_aro.py \
    --model-name your-model \
    --dataset Controlled_Images_A \
    --option four
```

---

## Common Issues and Solutions

### Issue 1: Out of Memory
**Solution:** Reduce `max_num_seqs` in `_get_vllm_config`:
```python
config = {
    "max_num_seqs": 2,  # Reduce from 5 to 2
}
```

### Issue 2: Wrong Output Format
**Solution:** Check your prompt template. Add debug prints:
```python
def _format_qwen_prompt(self, question):
    prompt_text = "..."
    print(f"DEBUG: Formatted prompt: {prompt_text}")  # Add this
    return prompt_text
```

### Issue 3: Slow Performance
**Diagnosis:** vLLM needs batches to be fast
- If `max_num_seqs=1`, it's not using batching
- Increase `max_num_seqs` if you have GPU memory

---

## Summary

The wrapper does three main things:

1. **Setup** (lines 11-46): Initialize vLLM with Qwen2-VL
2. **Format** (lines 64-94): Convert questions to Qwen's format
3. **Evaluate** (lines 96-281):
   - Load questions/answers
   - Batch images together
   - Generate with vLLM
   - Score outputs
   - Save results

The key insight is that vLLM handles all the complexity of batching and memory management - you just need to:
1. Format prompts correctly
2. Package them into vLLM's expected format
3. Call `llm.generate()`

---

## Next Steps

To understand this better, try:

1. **Add debug prints** to see the data flow:
```python
print(f"Batch size: {len(vllm_prompts)}")
print(f"First prompt: {vllm_prompts[0]['prompt'][:100]}")
```

2. **Run on a small dataset**:
```bash
python AdaptVis/main_aro.py --dataset Controlled_Images_A --option two
```

3. **Compare with standard transformers**:
- Time both implementations
- Check if outputs match

4. **Experiment with configs**:
- Try different `max_num_seqs` values
- Test different prompt formats
