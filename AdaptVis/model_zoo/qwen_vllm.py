import torch
import numpy as np
from tqdm import tqdm
import json
import os
import random
from PIL import Image
from vllm import LLM, SamplingParams


class QwenVLLMWrapper:
    """
    Wrapper for Qwen2-VL model using vLLM for faster inference.
    Based on the vLLM inference patterns used in scripts/vllm_inference.py
    """

    def __init__(
        self,
        root_dir,
        device,
        method="base",
        model_name="/leonardo_work/EUHPC_D27_102/compmech/models/Qwen2.5-3B",
    ):
        """
        Initialize Qwen2-VL model wrapper with vLLM.

        Args:
            root_dir: Directory for model cache
            device: Device to load model on (vLLM handles device placement)
            method: Evaluation method (not used for basic Qwen, kept for compatibility)
            model_name: Qwen model variant to use
        """
        # Get vLLM-specific model configuration
        vllm_config = self._get_vllm_config(model_name)

        # Initialize vLLM model
        print(f"Loading Qwen2-VL with vLLM: {model_name}")
        print(f"vLLM config: {vllm_config}")

        self.llm = LLM(model=model_name, **vllm_config)

        # Set sampling parameters for generation
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Greedy decoding for consistent results
            max_tokens=100,
            stop_token_ids=None,
        )

        self.device = device
        self.method = method
        self.model_name = model_name

    def _get_vllm_config(self, model_name):
        """Get vLLM-specific configuration for Qwen2-VL."""
        config = {
            "max_model_len": 4096,
            "max_num_seqs": 5,
            "limit_mm_per_prompt": {"image": 1},
        }

        # Add Qwen-specific processor kwargs
        config["mm_processor_kwargs"] = {
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }

        return config

    def _format_qwen_prompt(self, question):
        """
        Format question into Qwen2.5-VL chat template format for vLLM.
        Based on the format from scripts/vllm_inference.py
        """
        # Qwen2.5-VL format for vLLM
        prompt_text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt_text

    def _create_vllm_prompt(self, prompt, image):
        """
        Create a vLLM-compatible prompt dictionary.

        Args:
            prompt: Text prompt
            image: PIL Image

        Returns:
            Dictionary with prompt and multi_modal_data
        """
        formatted_prompt = self._format_qwen_prompt(prompt)

        vllm_prompt = {
            "prompt": formatted_prompt,
            "multi_modal_data": {"image": image.convert("RGB")},
        }
        return vllm_prompt

    @torch.no_grad()
    def get_out_scores_wh_batched(
        self,
        dataset,
        joint_loader,
        method,
        weight,
        option,
        threshold=None,
        weight1=None,
        weight2=None,
    ):
        """
        Generate outputs and scores for What's Up dataset using Qwen2-VL with vLLM.

        Args:
            dataset: Dataset name
            joint_loader: DataLoader with batched images
            method: Generation method (kept for compatibility)
            weight: Weight parameter (kept for compatibility)
            option: Number of options ('two', 'four', 'six')
            threshold: Threshold for adaptive methods
            weight1, weight2: Additional weights for adaptive methods

        Returns:
            Tuple of (scores, correct_ids)
        """
        scores = []
        index_of_total = 0
        acc = 0
        correct_id = []

        # Load prompts and answers
        qst_ans_file = f"prompts/{dataset}_with_answer_{option}_options.jsonl"

        with open(qst_ans_file, "r") as file:
            prompt_list = []
            answer_list = []
            for line in file:
                data = json.loads(line)
                prompt_list.append(data["question"])
                answer_list.append(data["answer"])

        # Sampling configuration
        SAMPLE = True
        TEST = os.getenv("TEST_MODE", "False") == "True"
        total_data_count = len(prompt_list)

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

            if TEST:
                all_indices = set(range(total_data_count))
                unsampled_indices = list(all_indices - set(sampled_indices))
                unsampled_indices.sort()
                sampled_indices = unsampled_indices

            prompt_list = [prompt_list[i] for i in sampled_indices]
            answer_list = [answer_list[i] for i in sampled_indices]

        results = []

        for batch in tqdm(joint_loader):
            batch_scores = []

            # Iterate over each image option in the batch
            for i_option in batch["image_options"]:
                im_scores = []

                # Collect all images and prompts for this batch to process together
                vllm_prompts = []
                for image in i_option:
                    prompt = prompt_list[index_of_total]
                    vllm_prompt = self._create_vllm_prompt(prompt, image)
                    vllm_prompts.append(vllm_prompt)
                    index_of_total += 1

                # Process all images in this option group with vLLM batching
                try:
                    outputs = self.llm.generate(vllm_prompts, self.sampling_params)

                    # Reset index for this batch
                    local_idx = 0
                    for output in outputs:
                        gen = output.outputs[0].text.strip()

                        # Get corresponding prompt and answer
                        batch_idx = index_of_total - len(i_option) + local_idx
                        prompt = prompt_list[batch_idx]
                        golden_answer = answer_list[batch_idx][0]

                        print(
                            f"Prompt: {prompt}\nGeneration: {gen}\nGolden: {golden_answer}"
                        )

                        result = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Golden": golden_answer,
                        }
                        results.append(result)

                        # Check if generation matches expected answer
                        c_option = batch["caption_options"]
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

                        elif len(list(c_option)) == 2:
                            if (
                                golden_answer in gen
                                or golden_answer.lower() in gen.lower()
                            ) and not (
                                golden_answer.lower() == "on"
                                and "front" in gen.strip().lower()
                            ):
                                acc += 1
                                correct_id.append(batch_idx)
                                answers = [1, 0]
                            else:
                                answers = [0, 1]

                        im_scores.append(np.expand_dims(np.array(answers), -1))
                        local_idx += 1

                except Exception as e:
                    print(f"Error during vLLM generation: {e}")
                    # Add dummy scores for failed batch
                    for _ in range(len(i_option)):
                        c_option = batch["caption_options"]
                        if len(list(c_option)) == 4:
                            answers = [0, 0, 1, 0]
                        else:
                            answers = [0, 1]
                        im_scores.append(np.expand_dims(np.array(answers), -1))

                batch_scores.append(np.concatenate(im_scores, axis=-1))

            scores.append(batch_scores)

            # Save results periodically
            output_file_path = f"./output/results_qwen_vllm_{dataset}_{method}_{option}option_{TEST}.json"
            with open(output_file_path, "w", encoding="utf-8") as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)

            print(
                f"Accuracy: {acc}/{index_of_total} = {acc / index_of_total if index_of_total > 0 else 0}"
            )

        # Save final scores
        if index_of_total > 0:
            print(f"Final Accuracy: {acc / index_of_total}")
            output_score_file = output_file_path.replace(".json", "scores.json")
            with open(output_score_file, "w", encoding="utf-8") as fout:
                json.dump(
                    {"acc": acc / index_of_total, "correct_id": correct_id},
                    fout,
                    ensure_ascii=False,
                    indent=4,
                )

        # Return scores
        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        if dataset in ["Controlled_Images_B", "Controlled_Images_A"]:
            return (all_scores, [])
        else:
            return (acc / index_of_total if index_of_total > 0 else 0, correct_id)
