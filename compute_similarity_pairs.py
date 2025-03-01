import argparse
import json
import multiprocessing as mp
import os
import random

import tqdm
from anthropic import Anthropic
from openai import OpenAI

from api_keys import ANTHROPIC_API_KEY, OPENAI_API_KEY
from grade_outputs_parallel import (
    FOLLOWUP_PROMPT_DATASET,
    MODEL_TO_MODEL,
    FollowupPrompt,
    query_model_api,
)
from utils import get_datetime_str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--followup-prompts", type=str, required=True)
    parser.add_argument("--negate", action="store_true")
    return parser.parse_args()


def get_followup_prompt(followup_prompts: list[FollowupPrompt], modifier: str) -> str:
    try:
        return next(filter(lambda x: x.modifier == modifier, followup_prompts))
    except StopIteration:
        print(f"Could not find followup prompt for {modifier}")
        raise ValueError


def create(
    rank: int,
    followup_prompts: list[FollowupPrompt],
    client: OpenAI | Anthropic,
    model_name: str,
    pairs: list[tuple[str, str]],
    results_queue: mp.Queue,
):
    results = []
    for i in tqdm.trange(len(pairs), desc=f"Rank {rank}"):
        mod_1, mod_2 = pairs[i]
        try:
            mod_1_prompt = get_followup_prompt(followup_prompts, mod_1).eval_prompt
            mod_2_prompt = get_followup_prompt(followup_prompts, mod_2).eval_prompt
        except StopIteration:
            continue

        if args.negate:
            prompt = f"If a smart person edited text so it {mod_2_prompt}, will they almost always produce text that {mod_1_prompt}? Think step-by-step, breifly explain your thought process, and then respond with YES or NO, in all caps."
        else:
            prompt = f"If a smart person edited text so it {mod_2_prompt}, will they sometimes produce text that {mod_1_prompt}? Think step-by-step, breifly explain your thought process, and then respond with YES or NO."

        try:
            response = query_model_api(
                client, model_name, [{"role": "user", "content": prompt}]
            )
        except Exception:
            response = None

        results.append((mod_1, mod_2, response))
        # print(results[-1])

    results_queue.put(results)


if __name__ == "__main__":
    args = parse_args()

    client, model_name = MODEL_TO_MODEL[args.model]
    followup_prompts = FOLLOWUP_PROMPT_DATASET[args.followup_prompts]

    pairs_list = []
    for i in range(len(followup_prompts)):
        for j in range(len(followup_prompts)):
            if (
                i == j
                or followup_prompts[i].modifier == ""
                or followup_prompts[j].modifier == ""
            ):
                continue
            pairs_list.append(
                (
                    followup_prompts[i].modifier,
                    followup_prompts[j].modifier,
                )
            )

    random.shuffle(pairs_list)
    # pairs_list = pairs_list[:100]

    print(len(pairs_list), "pairs to compute")

    if client == "openai":
        client = OpenAI(api_key=OPENAI_API_KEY)
    elif client == "anthropic":
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
    else:
        raise ValueError(f"Unknown client {client}")

    print("loaded model", model_name)

    print("generating dataset")
    N_THREADS = min(500, len(pairs_list))
    results_queue = mp.Queue(N_THREADS)
    n_per_thread = len(pairs_list) // N_THREADS
    remainder = len(pairs_list) % N_THREADS
    processes = []
    start = 0
    for rank in range(N_THREADS):
        end = start + n_per_thread
        if rank < remainder:
            end += 1

        if rank == N_THREADS - 1:
            end = len(pairs_list)

        print(f"Starting {rank + 1} of {N_THREADS}")
        p = mp.Process(
            target=create,
            args=(
                rank,
                followup_prompts,
                client,
                model_name,
                pairs_list[start:end],
                results_queue,
            ),
        )
        processes.append(p)

        start = end

    for p in processes:
        p.start()

    final_results = []
    for i in range(len(processes)):
        final_results.extend(results_queue.get())
        print("got", len(final_results), "results out of", len(pairs_list))

    filepath = os.path.join(
        "similarity_pairs",
        f"{args.model}_{args.followup_prompts}_{get_datetime_str()}{'_negate' if args.negate else ''}.json",
    )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print("saving to", filepath)
    with open(filepath, "w") as f:
        json.dump(final_results, f)

    with open("similarity_pairs/negate_dump.json", "w") as f:
        json.dump(final_results, f)

    print("saved")

    YES = 0
    NO = 0
    for i in range(len(final_results)):
        if final_results[i][2] is None:
            continue
        if "YES" in final_results[i][2]:
            YES += 1
        else:
            NO += 1

    print(YES, NO, YES / (YES + NO), len(final_results) - YES - NO)

    for p in processes:
        p.join()
