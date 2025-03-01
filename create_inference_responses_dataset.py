import argparse
import os
import pickle
import random

import torch.multiprocessing as mp

from data.followup_prompts import FOLLOWUP_PROMPT_DATASET, FollowupPrompt
from data.inference_prompts import INFERENCE_PROMPTS
from multiprocess_utils import gen_batch, load_chatllm
from utils import get_datetime_str
from utils.llm import ChatLLM, Message


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--infer-prompts", type=str, required=True)
    parser.add_argument("--followup-prompts", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="datasets")
    return parser.parse_args()


def get_indefinite_article(word: str):
    vowels = "aeiou"
    special_cases = {
        "hour": "an",
        "honest": "an",
        "herb": "a",  # American English
        "university": "a",
        "unicorn": "a",
        "one-time": "a",
        "euler": "a",
        "european": "a",
    }

    if word.lower() in special_cases:
        return special_cases[word.lower()]

    if word[0].lower() in vowels:
        return "an"
    else:
        return "a"


def create(
    llm: ChatLLM,
    model: str,
    user_queries: list[str],
    followup_prompts: list[FollowupPrompt],
) -> list:
    queries = []
    for i in range(len(user_queries)):
        for j in range(len(followup_prompts)):
            modifier = followup_prompts[j].modifier
            first_word = user_queries[i].split(" ")[0]
            if len(modifier) == 0:  # control
                prompt = f"Write {get_indefinite_article(first_word)} {user_queries[i]}"
            else:
                prompt = f"Write {get_indefinite_article(modifier)} {modifier} {user_queries[i]} Ensure the {first_word} is maximally {modifier}."

            queries.append(
                {"user_query": user_queries[i], "prompt": prompt, "modifier": modifier}
            )

    print(random.choice(queries)["prompt"])

    full_prompts = [llm.chat_into_str([Message("user", q["prompt"])]) for q in queries]

    print(random.choice(full_prompts))

    # outputs = llm.generate_batch(
    #     full_prompts,
    #     use_tqdm=True,
    # )
    outputs = gen_batch(model, llm, full_prompts, max_tokens=10000)

    for i in range(len(queries)):
        queries[i]["output"] = outputs[i]

    return queries


if __name__ == "__main__":
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    infer_prompts = INFERENCE_PROMPTS[args.infer_prompts]
    followup_prompts = FOLLOWUP_PROMPT_DATASET[args.followup_prompts]

    print("loading model")

    llm = load_chatllm(args.model)

    print("loaded model " + llm.model_name)
    print("generating responses")

    results = create(llm, args.model, infer_prompts, followup_prompts)

    folder_path = f"inference_responses_{args.model}_{args.infer_prompts}_{args.followup_prompts}_{get_datetime_str()}"

    print("saving results to " + folder_path)

    file_path = os.path.join(args.output_dir, folder_path, "inference_generated.pkl")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(results, f)
