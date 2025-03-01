import argparse
import json
import os
import pickle

import torch.multiprocessing as mp

from data.followup_prompts import FOLLOWUP_PROMPT_DATASET, FollowupPrompt
from multiprocess_utils import gen_batch, load_chatllm
from utils import get_datetime_str
from utils.llm import ChatLLM, Message


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--followup", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    return parser.parse_args()


def generate_prompt(question: str, resp: str, followup: FollowupPrompt) -> str:
    return f"""The following RESPONSE is a response to question QUESTION:
QUESTION {question}
RESPONSE {resp}

{followup.prompt}"""


def create(
    llm: ChatLLM, model: str, user_queries: list[str], followups: list[FollowupPrompt]
) -> list:
    print("Generating general responses")
    full_prompts = [[Message("user", query)] for query in user_queries]
    outputs = llm.generate_batch(
        [llm.chat_into_str(prompt) for prompt in full_prompts],
        use_tqdm=True,
    )

    results = []
    for i in range(len(user_queries)):
        for j in range(len(followups)):
            # llama3 starts with this

            prompt = [
                Message(
                    "user", generate_prompt(user_queries[i], outputs[i], followups[j])
                ),
            ]

            prompt_str = llm.chat_into_str(prompt)

            if "Llama-3" in llm.model_name and "Instruct" in llm.model_name:
                prompt_str += "Here is the revised RESPONSE"

            results.append(
                {
                    "user_query": user_queries[i],
                    "query": prompt_str,
                    "modifier": followups[j].modifier,
                    "control_output": outputs[i],
                    # "output": outputs[i],
                    "query_start": prompt_str.find(followups[j].prompt),
                }
            )

    print(results[3]["query"])

    print("Generating followup responses")
    outputs = gen_batch(
        model, llm, [each["query"] for each in results], max_tokens=3000
    )

    for i in range(len(results)):
        if "Llama-3" in llm.model_name and "Instruct" in llm.model_name:
            outputs[i] = "<|start_header_id|>assistant<|end_header_id|>\n" + "\n".join(
                outputs[i].split("\n")[1:]
            )

        print(outputs[i])
        results[i]["output"] = outputs[i]

    return results


if __name__ == "__main__":
    args = parse_args()
    mp.set_start_method("spawn")

    user_queries = json.load(
        open(os.path.join("user_queries", args.dataset + ".json"), "r")
    )

    followups = FOLLOWUP_PROMPT_DATASET[args.followup]

    print("Loading model")
    llm = load_chatllm(args.model)
    print("Loaded model " + llm.model_name)

    print("Generating dataset")

    results = create(llm, args.model, user_queries, followups)

    folder_path = (
        f"user_query_{args.model}_{args.dataset}_{args.followup}_{get_datetime_str()}"
    )

    print("Saving", folder_path)

    file_path = os.path.join(
        args.output_dir,
        folder_path,
        "generated.pkl",
    )

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(results, f)
