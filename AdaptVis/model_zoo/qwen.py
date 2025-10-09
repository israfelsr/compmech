import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
import random
from PIL import Image


class QwenWrapper:
    """
    Wrapper for Qwen2-VL model to work with the AdaptVis evaluation framework.
    """

    def __init__(self, root_dir, device, method='base', model_name='Qwen/Qwen2-VL-7B-Instruct'):
        """
        Initialize Qwen2-VL model wrapper.

        Args:
            root_dir: Directory for model cache
            device: Device to load model on
            method: Evaluation method (not used for basic Qwen, kept for compatibility)
            model_name: Qwen model variant to use
        """
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=root_dir,
            device_map="auto" if device == "cuda" else device
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=root_dir
        )

        self.device = device
        self.method = method

    def _format_qwen_messages(self, prompt, image):
        """
        Format prompt and image into Qwen2-VL message format.

        Args:
            prompt: Text prompt
            image: PIL Image

        Returns:
            List of messages in Qwen format
        """
        # Convert PIL image to format expected by Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages

    @torch.no_grad()
    def get_out_scores_wh_batched(self, dataset, joint_loader, method, weight, option, threshold=None, weight1=None, weight2=None):
        """
        Generate outputs and scores for What's Up dataset using Qwen2-VL.

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

        # Map datasets to their available option files
        dataset_options = {
            "COCO_QA_one_obj": "four",
            "COCO_QA_two_obj": "four",
            "Controlled_Images_A": "four",
            "Controlled_Images_B": "four",
            "VG_QA_one_obj": "six",
            "VG_QA_two_obj": "six",
        }

        # Use dataset-specific option if available, otherwise use provided option
        actual_option = dataset_options.get(dataset, option)

        # Load prompts and answers
        qst_ans_file = f'prompts/{dataset}_with_answer_{actual_option}_options.jsonl'

        with open(qst_ans_file, 'r') as file:
            prompt_list = []
            answer_list = []
            for line in file:
                data = json.loads(line)
                prompt_list.append(data["question"])
                answer_list.append(data["answer"])

        # Sampling configuration
        SAMPLE = True
        TEST = os.getenv('TEST_MODE', 'False') == 'True'
        total_data_count = len(prompt_list)

        if SAMPLE:
            idx_file_path = f'./output/sampled_idx_{dataset}.npy'

            if os.path.exists(idx_file_path):
                sampled_indices = np.load(idx_file_path).tolist()
            else:
                sampled_indices = random.sample(range(total_data_count), int(0.2 * total_data_count))
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

                for image in i_option:
                    prompt = prompt_list[index_of_total]

                    # Format messages for Qwen
                    messages = self._format_qwen_messages(prompt, image)

                    # Prepare for inference
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)

                    # Generate
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=100
                    )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    gen = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]

                    print(f"Prompt: {prompt}\\nGeneration: {gen}\\nGolden: {answer_list[index_of_total][0]}")

                    result = {
                        "Prompt": prompt,
                        "Generation": gen,
                        "Golden": answer_list[index_of_total][0],
                    }
                    results.append(result)

                    # Check if generation matches expected answer
                    c_option = batch["caption_options"]
                    if len(list(c_option)) == 4:
                        if (answer_list[index_of_total][0] in gen or answer_list[index_of_total][0].lower() in gen.lower()) \\
                                and not (answer_list[index_of_total][0].lower() == 'on' and 'front' in gen.strip().lower()):
                            acc += 1
                            correct_id.append(index_of_total)
                            answers = [1, 0, 0, 0]
                        else:
                            answers = [0, 0, 1, 0]

                    elif len(list(c_option)) == 2:
                        if (answer_list[index_of_total][0] in gen or answer_list[index_of_total][0].lower() in gen.lower()) \\
                                and not (answer_list[index_of_total][0].lower() == 'on' and 'front' in gen.strip().lower()):
                            acc += 1
                            correct_id.append(index_of_total)
                            answers = [1, 0]
                        else:
                            answers = [0, 1]

                    im_scores.append(np.expand_dims(np.array(answers), -1))
                    index_of_total += 1

                batch_scores.append(np.concatenate(im_scores, axis=-1))

            scores.append(batch_scores)

            # Save results
            output_file_path = f'./output/results_qwen_{dataset}_{method}_{option}option_{TEST}.json'
            with open(output_file_path, 'w', encoding='utf-8') as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)
            print(f"Accuracy: {acc}/{index_of_total} = {acc / index_of_total if index_of_total > 0 else 0}")

        # Save final scores
        if index_of_total > 0:
            print(f"Final Accuracy: {acc / index_of_total}")
            output_score_file = output_file_path.replace(".json", "scores.json")
            with open(output_score_file, 'w', encoding='utf-8') as fout:
                json.dump({"acc": acc / index_of_total, "correct_id": correct_id}, fout, ensure_ascii=False, indent=4)

        # Return scores
        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        if dataset in ['Controlled_Images_B', 'Controlled_Images_A']:
            return (all_scores, [])
        else:
            return (acc / index_of_total if index_of_total > 0 else 0, correct_id)
