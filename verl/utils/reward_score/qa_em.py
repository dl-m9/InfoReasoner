# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import string
import random
import sys
import os
import torch
from verl import DataProto
import numpy as np
from typing import Union, Optional
import threading

# Thread-local storage for reward metrics
_reward_metrics = threading.local()

# Thread-local storage for validation flag
_validation_flag = threading.local()


# Ensure project root is importable (for IG package)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from IG.calculate import (  # noqa: E402
    calculate_uncertainty_soft_batch,
    create_collate_fn,
    gen_answers_batch,
    process_item_for_ig,
)
from IG.models.huggingface_models import HuggingfaceModel  # noqa: E402
from IG.uncertainty_measures.semantic_entropy import EntailmentDeberta  # noqa: E402

# Try to import IG client (optional)
try:
    from verl.utils.reward_score.IG_client import get_ig_client, is_ig_service_available
    IG_CLIENT_AVAILABLE = True
except ImportError:
    IG_CLIENT_AVAILABLE = False
    get_ig_client = None
    is_ig_service_available = None


def save_question_infomation_ground_truth(
    question: str,
    information_blocks,
    ground_truth,
    save_path: str = "/forest/forest/Search-R1-info/script_my/info_gain_group_study/info_gain_group_study_items.json",
) -> None:
    """Persist question, context, and answers for info-gain group study.

    Each call appends ONE item to a JSON list stored on disk.
    If the file doesn't exist, it will be created with a list; if it exists, it will
    be loaded, appended, and written back.
    """
    try:
        # Normalize inputs
        if information_blocks is None:
            information_blocks = []
        if ground_truth is None:
            ground_truth = []

        # Build context string
        if isinstance(information_blocks, (list, tuple)):
            context = "\n".join([str(b).strip() for b in information_blocks if b is not None])
        else:
            context = str(information_blocks)

        # Normalize answers to list of strings
        if isinstance(ground_truth, str):
            answers = [ground_truth]
        elif isinstance(ground_truth, (list, tuple)):
            answers = list(ground_truth)
        else:
            answers = [str(ground_truth)]

        item = {
            "question": question,
            "context": context,
            "answers": answers,
        }

        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Load existing list if file exists
        existing = []
        if os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        existing = json.loads(content)
            except Exception:
                # If file is corrupted or not a JSON list, start fresh
                existing = []

        existing.append(item)

        # Rewrite full list
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False)
    except Exception as e:
        print(f"[WARNING] save_question_infomation_ground_truth failed: {e}")


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are no matches, return None
    if len(matches) == 0:
        return None

    # Return the last answer (for consistency with original logic)
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1., info_gain_score=None):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        info_gain_score: Optional pre-computed info gain score. If None, will compute it.
    """
    result = extract_question_information(solution_str=solution_str)
    question = result['question']
    information_blocks = result['information_blocks']
    answer = extract_solution(solution_str=solution_str)

    # save_question_infomation_ground_truth(
    #     question=question,
    #     information_blocks=information_blocks,
    #     ground_truth=ground_truth.get('target', []),
    # )
    
    
    do_print = random.randint(1, 64) == 1
    # do_print = True  # Uncomment to always print
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers 黄金答案: {ground_truth['target']}")
        print(f"Extracted answer 提取的答案: {answer}")
        print(f"Solution string 解决方案字符串: {solution_str}")
        print(f"--------------------------------")
    
    # Check if we're in validation mode (skip info gain computation during validation)
    is_validation = getattr(_validation_flag, 'skip_info_gain', False)
    # em_score_weight = 0.0
    
    # optional: compute information gain if handles are provided and we have retrieval
    info_gain_score_weight = 0.6
    if len(information_blocks) == 0:
        info_gain_score_weight = 0.0
    if is_validation:
        info_gain_score_weight = 0.0  # Skip info gain during validation
        # em_score_weight = 1.0
    

    
    # Use pre-computed info_gain_score if provided, otherwise compute it
    if info_gain_score is None:
        info_gain_score = 0.0
        debug_print = False
        if debug_print:
            print(f"[DEBUG] IG_CLIENT_AVAILABLE: {IG_CLIENT_AVAILABLE}")
            print(f"[DEBUG] len(information_blocks): {len(information_blocks)}")
            print(f"[DEBUG] question: {question}")
            print(f"[DEBUG] information_blocks: {information_blocks}")
        # Try IG service first if available
        # Skip info gain computation during validation for efficiency
        if not is_validation and IG_CLIENT_AVAILABLE and len(information_blocks) > 0:
            if debug_print:
                print(f"[DEBUG] Attempting to get IG client...")
            # Use environment variable if set, otherwise use default
            service_url = os.getenv('IG_SERVICE_URL', 'http://0.0.0.0:0310')
            ig_client = get_ig_client(service_url=service_url)
            if debug_print:
                print(f"[DEBUG] ig_client: {ig_client}")
                print(f"[DEBUG] service_url: {service_url}")
            if ig_client is not None:
                if debug_print:
                    print(f"[DEBUG] Checking service availability...")
                # Use the client's health_check method directly instead of is_ig_service_available()
                service_available = ig_client.health_check()
                if debug_print:
                    print(f"[DEBUG] ig_client.health_check(): {service_available}")
                if service_available:
                    try:
                        context = "\n".join(information_blocks)
                        if debug_print:
                            print(f"[DEBUG] Calling compute_info_gain with question: {question[:50]}..., context length: {len(context)}")
                            print(f"[DEBUG] ground_truth['target'] type: {type(ground_truth['target'])}, value: {ground_truth['target']}")
                        info_gain_score = ig_client.compute_info_gain(
                            question=question,
                            context=context,
                            answers=ground_truth['target']  # _normalize_answers will handle all cases
                        )
                        if debug_print:
                            print(f"[DEBUG] compute_info_gain returned: {info_gain_score}")
                    except Exception as e:
                        print(f"[WARNING] IG service call failed, falling back to local computation: {e}")
                        import traceback
                        traceback.print_exc()
                        info_gain_score = 0.0
                else:
                    print(f"[DEBUG] Service health check failed, skipping IG computation")
                    print(f"[DEBUG] Service URL: {ig_client.service_url}")
            else:
                print(f"[DEBUG] get_ig_client() returned None")
    else:
        # Use pre-computed info_gain_score
        pass
    
    

    
    if answer is None:
        output_score = 0
        all_score = 0
    else:
        if em_check(answer, ground_truth['target']):
            
            output_score = score  # Only EM check score
            all_score = score + info_gain_score_weight * info_gain_score  # Weighted sum
        else:
            output_score = format_score  # Only EM check score
            all_score = info_gain_score_weight * info_gain_score + format_score  # Weighted sum
    
    if do_print:
        print("--------------------------------")
        print("info_gain_score: ", info_gain_score)
        print("output_score: ", output_score)
        print("all_score: ", all_score)
    # Store metrics in thread-local storage
    _reward_metrics.info_gain_score = info_gain_score
    _reward_metrics.output_score = output_score  # Only EM check score
    _reward_metrics.all_score = all_score  # Weighted sum
    
    return all_score  # Return weighted sum for training


def get_last_reward_metrics():
    """Get reward metrics from the last compute_score_em call.
    
    Returns:
        dict with 'info_gain_score', 'output_score' (EM check only), and 'all_score' (weighted sum)
    """
    try:
        return {
            'info_gain_score': getattr(_reward_metrics, 'info_gain_score', 0.0),
            'output_score': getattr(_reward_metrics, 'output_score', 0.0),  # EM check only
            'all_score': getattr(_reward_metrics, 'all_score', 0.0),  # Weighted sum
        }
    except:
        return {'info_gain_score': 0.0, 'output_score': 0.0, 'all_score': 0.0}



def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1., *, generator=None, entailment_model=None, device=None):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    result = extract_question_information(solution_str=solution_str)
    question = result['question']
    information_blocks = result['information_blocks']
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"--------------------------------")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def extract_question_information(solution_str):
    """Extract question and retrieved information from a solution string.

    Returns a dict containing:
    - question: the question text parsed from the user turn (after "Question:")
    - retrieved_information: concatenated string of all <information>...</information> blocks
    - information_blocks: list of individual information blocks
    """
    # 1) Try to locate the user block and pull the text after "Question:"
    question = None
    user_block_match = re.search(r"<\|im_start\|>user(.*?)<\|im_end\|>", solution_str, re.DOTALL | re.IGNORECASE)
    if user_block_match:
        user_block = user_block_match.group(1)
        q_match = re.search(r"Question:\s*(.*?)(?:\n|$)", user_block, re.IGNORECASE | re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()
    else:
        # Fallback: search globally for a "Question:" field
        q_match = re.search(r"Question:\s*(.*?)(?:\n|<\|im_end\|>|$)", solution_str, re.IGNORECASE | re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()

    # 2) Extract retrieved information blocks
    info_pattern = r"<information>(.*?)</information>"
    information_blocks = [s.strip() for s in re.findall(info_pattern, solution_str, re.DOTALL)]
    
    # Filter out prompt-related text: remove "and" that comes from prompt "between <information> and </information>"
    filtered_blocks = []
    for block in information_blocks:
        # Remove leading/trailing "and" that might come from the prompt
        filtered_block = block.strip()
        # Skip if the block is just "and" (from prompt text)
        if filtered_block.lower() == 'and':
            continue
        
        if filtered_block:  # Only add non-empty blocks
            filtered_blocks.append(filtered_block)
    
    information_blocks = filtered_blocks
    retrieved_information = "\n".join(information_blocks) if information_blocks else ""

    return {
        'question': question,
        'retrieved_information': retrieved_information,
        'information_blocks': information_blocks,
    }

if __name__ == "__main__":
    solution_str = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: what type of monster is zoe from super monsters?
<|im_end|>
<|im_start|>assistant
 <think> To answer this question, I need to determine the type of monster Zoe belongs to in the 'Super Monsters' series. To do that, I need to recall information about Zoe and her role within the context of the Super Monsters universe. </think>
<search> What type of monster is Zoe from Super Monsters? </search>
 <information>Doc 1(Title: "Gamera: Super Monster") Gamera: Super Monster When the evil alien Zanon comes to enslave Earth, all hope seems lost. The Earth's resident superheroes, the Spacewomen, are powerless to stop him. They must enlist the help of a young boy who has a special connection with the giant turtle Gamera. The Friend of All Children then battles Gyaos (a huge vampire bat/pterosaur hybrid), Zigra (an alien shark), Viras (an alien squid), Jiger (a female giant prehistoric dinosaur), Guiron (a knife-headed alien monster) and finally Barugon (an enormous lizard whose tongue sprays a freeze-gas that can freeze things solid and whose back spines emit a
 Doc 2(Title: "Gamera: Super Monster") the uncut Japanese versions on DVD for the first time in North America. These ""Special Edition"" DVDs were released in sequential order, starting with " (1965) on May 18, 2010. Gamera: Super Monster When the evil alien Zanon comes to enslave Earth, all hope seems lost. The Earth's resident superheroes, the Spacewomen, are powerless to stop him. They must enlist the help of a young boy who has a special connection with the giant turtle Gamera. The Friend of All Children then battles Gyaos (a huge vampire bat/pterosaur hybrid), Zigra (an alien shark), Viras (an alien squid), Jiger (a female
 Doc 3(Title: "Zoé Kézako") Zoé Kézako Zoé Kézako is a French animated television series, adapted from Véronique Saüquère's book series published by Frimousse. The first season was produced by Corinne Kouper at Sparkling; the second season was produced at TeamTO. This second season is currently airing in France on TF1 and Télétoon. The series is also broadcast around the world: in Norway (NRK1), Sweden (SVT), Denmark (DR), Italy (Rai 3), Canada (Radio Canada), Belgium (RTBF and Ketnet), Portugal (RTP), Finland (YLE), Ireland (RTÉ) and in Argentina (Pakapaka). A big round face with two springy pigtails on top, and a gangly mouth with a playful grin which, if turned more serious, almost turns out
 </information>
 <think> From the provided information, I could not find detailed information about Zoe's specific type of monster. However, I noticed a series titled 'Super Monsters,' thus I need to assume Zoe comes from this franchise. Upon further recall, I found that she is part of the Lunar league along with Flicka, Rockett and Slacker. </think>
 <search> What is the Lunar league in the Super Monsters franchise? </search>
 
 <information>Doc 1(Title: "Lunar: The Silver Star") next Dragonmaster, and save the world in the process. ""Lunar: The Silver Star"" was critically and commercially successful, becoming the number one selling Mega-CD title in Japan and the second highest-selling Mega-CD title of all time. As the first game in the ""Lunar"" series, it set the standard for other follow-up titles including the direct sequel " in 1994. Since the game's original release, three enhanced remakes have been produced for various systems: " in 1996, ""Lunar Legend"" in 2002, and " in 2009. ""Lunar: The Silver Star"" is a traditional, top-down role-playing video game featuring two-dimensional characters and environments.
 Doc 2(Title: "Lunar: Sanposuru Gakuen") Lunar: Sanposuru Gakuen Even though the game was released after "", it is set hundreds of years before it. Several creatures and places from the game also make an appearance. The game was remade two years later for the Sega Saturn as Mahō Gakuen Lunar!. The game has several elements common to role-playing video games, with 2D character sprites and environments. During the game, the player will encounter monsters that are fought randomly within dungeons, and gain experience from winning battles. Magic and additional skills are unlocked through leveling up, as well as allowing the characters to grow stronger. The
 Doc 3(Title: "Lunar Magic") Lunar Magic Lunar Magic is a level editor created by FuSoYa for ""Super Mario World"". The interface runs on Microsoft Windows and allows people to create custom graphics and edit levels and the world map. According to FuSoYa, he started the development of Lunar Magic for the Super Nintendo Entertainment System game ""Super Mario World"", in February 2000. The first public release was on September 24, 2000. New releases with improvements continued over the years. It was last updated on February 9, 2018 to version 2.53. Reception for ""Lunar Magic"" has been positive. Kotaku praised it, noting it was a</information>
 
 <think> From the Lunar series information provided, Zoe is part of the 'Lunar league,' which consists of Flicka, Rockett and Slacker. Given this context I believe Zoe's type of monster must be from the Super Monsters franchise instead of the Lunar franchise. Therefore, Zoe appears to be a Super Monster as it follows up on previous types of monsters. </think>
 <answer> Super monster </answer>
"""
    result = extract_question_information(solution_str)
    question = result['question']
    information_blocks = result['information_blocks']
    print(f"[DEBUG] question: {question}")
    print(f"[DEBUG] information_blocks: {information_blocks}")
    print(f"[DEBUG] information_blocks type: {type(information_blocks)}")
    print(f"[DEBUG] information_blocks length: {len(information_blocks)}")