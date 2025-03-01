import argparse
import json
import os
import pickle
import random

import torch.multiprocessing as mp
import tqdm
from anthropic import Anthropic
from openai import OpenAI

from api_keys import ANTHROPIC_API_KEY, OPENAI_API_KEY
from compute_grads import consolidate_by_query
from data.followup_prompts import FOLLOWUP_PROMPT_DATASET, FollowupPrompt
from utils import get_datetime_str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modifiers", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt3")
    parser.add_argument("--followup-prompts", type=str, required=True)
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--cutoff-modifiers", type=int, default=None)
    return parser.parse_args()


MODEL_TO_MODEL = {
    "gpt3": ("openai", "gpt-3.5-turbo-0125"),
    "gpt4": ("openai", "gpt-4-turbo-2024-04-09"),
    "gpt4o": ("openai", "gpt-4o-2024-05-13"),
    "sonnet": ("anthropic", "claude-3-sonnet-20240229"),
    "haiku": ("anthropic", "claude-3-haiku-20240307	"),
    "opus": ("anthropic", "claude-3-opus-20240229"),
}


def get_followup_prompt(followup_prompts: list[FollowupPrompt], modifier: str) -> str:
    try:
        return next(filter(lambda x: x.modifier == modifier, followup_prompts))
    except StopIteration:
        print(f"Could not find followup prompt for {modifier}")
        raise ValueError


def query_model_api(
    client: OpenAI | Anthropic,
    model_name: str,
    messages: list,
    max_output_tokens: int = 1000,
) -> str:
    if isinstance(client, OpenAI):
        completion = client.chat.completions.create(
            model=model_name,
            max_tokens=max_output_tokens,
            messages=messages,
        )
        return completion.choices[0].message.content
    elif isinstance(client, Anthropic):
        completion = client.messages.create(
            model=model_name,
            max_tokens=max_output_tokens,
            messages=messages,
        )
        return completion.content[0].text
    else:
        raise ValueError("Invalid client")


def create(
    client: OpenAI | Anthropic,
    model_name: str,
    mod_eval: str,
    mod_indep: str,
    all_queries: list,
    eval_prompt: str,
    cutoff: int | None,
    queue: mp.Queue,
) -> list:
    results = []

    if eval_prompt == "":
        raise ValueError("Eval prompt is empty")

    prompt = f"I have two responses to the same question. Please tell me which response, A or B, {eval_prompt}. Think step-by-step and use evidence to reason. Then, write THE ANSWER IS: A or B."

    queries = [q for q in all_queries if q["modifier"] == mod_indep]

    if len(queries) == 0:
        print("No queries for " + mod_indep)
        return results

    print("Starting", mod_eval, mod_indep)

    if cutoff:
        queries = queries[: args.cutoff]

    for i in tqdm.trange(len(queries), desc=f"{mod_eval} {mod_indep}"):
        if "<|start_header_id|>assistant<|end_header_id|>" in queries[i]["output"]:
            # print("Skipping", queries[i]["output"][:1])
            assert queries[i]["output"].startswith(
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            ), json.dumps(queries[i], indent=2)
            queries[i]["output"] = queries[i]["output"].split(
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )[1]
        if random.random() > 0.5:
            a_resp = queries[i]["control_output"]
            b_resp = queries[i]["output"]
            a_control = True
        else:
            a_resp = queries[i]["output"]
            b_resp = queries[i]["control_output"]
            a_control = False

        full_prompt = prompt + f"\n\nRESPONSE A:\n{a_resp}\n\nRESPONSE B:\n{b_resp}"

        try:
            response = query_model_api(
                client, model_name, [{"role": "user", "content": full_prompt}]
            )

            results.append(
                {
                    "control_output": queries[i]["control_output"],
                    "output": queries[i]["output"],
                    "a_control": a_control,
                    "prompt": full_prompt,
                    "response": response,
                }
            )

            # print(mod_eval, mod_indep, json.dumps(results[-1], indent=2))
        except Exception as e:
            print(e)
            # print("Failed", full_prompt)

    queue.put((mod_eval, mod_indep, results))


if __name__ == "__main__":
    args = parse_args()

    model_info = MODEL_TO_MODEL[args.model]
    followup_prompts = FOLLOWUP_PROMPT_DATASET[args.followup_prompts]

    assert (
        args.followup_prompts in args.dataset
    ), f"Expected {args.followup_prompts} in {args.dataset}"

    with open(args.dataset, "rb") as f:
        queries = pickle.load(f)

    if "inference_generated.pkl" in args.dataset:
        # need to add control_output
        by_query = consolidate_by_query(queries)

        for q in queries:
            q["control_output"] = by_query[q["user_query"]][""]["output"]

    with open(args.modifiers, "r") as f:
        modifiers = json.load(f)  # [[a, b], [a, b], ...]

    random.shuffle(modifiers)

    if args.cutoff_modifiers:
        modifiers = modifiers[: args.cutoff_modifiers]

    print("running", modifiers)

    if model_info[0] == "openai":
        llm = OpenAI(api_key=OPENAI_API_KEY)
    elif model_info[0] == "anthropic":
        llm = Anthropic(api_key=ANTHROPIC_API_KEY)
    else:
        raise ValueError("Invalid model: " + args.model)
    print("Loaded model " + model_info[1])

    print("Generating dataset")
    results_queue = mp.Queue(len(modifiers))
    processes = []
    for i in range(len(modifiers)):
        mod_eval, mod_indep = modifiers[i][:2]

        p = mp.Process(
            target=create,
            args=(
                llm,
                model_info[1],
                mod_eval,
                mod_indep,
                queries,
                get_followup_prompt(followup_prompts, mod_eval).eval_prompt,
                args.cutoff,
                results_queue,
            ),
        )
        processes.append(p)

    print("Waiting for processes to finish")

    result_dict = {}

    # processes[0].start()
    for i in range(len(processes)):
        processes[i].start()

    for i in range(len(processes)):
        # if i + 1 < len(processes):
        # processes[i + 1].start()
        mod_eval, mod_indep, results = results_queue.get()
        print("Got results for", (mod_eval, mod_indep))
        result_dict[(mod_eval, mod_indep)] = results

    print("All processes finished")

    folder_path = os.path.join(
        "comparisons",
        f"{args.model}_{get_datetime_str()}",
    )
    os.makedirs(folder_path, exist_ok=True)

    print("Saving to", folder_path)

    with open(os.path.join(folder_path, "results.pkl"), "wb") as f:
        pickle.dump(
            {
                "dataset": args.dataset,
                "modifiers": modifiers,
                "results": result_dict,
                "model": model_info[1],
                "modifiers_file": args.modifiers,
            },
            f,
        )
    print("saved")

    for p in processes:
        p.join()
