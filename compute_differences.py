import argparse
import datetime
import json
import os
import pickle
import random

import numpy as np
import pandas as pd

from data.followup_prompts import FOLLOWUP_PROMPT_DATASET, FollowupPrompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--grad-dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--followup-prompts", type=str, required=True)
    parser.add_argument("--mturk-dir", type=str, default="mturk")
    parser.add_argument("--negate", action="store_true")

    return parser.parse_args()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_followup_prompt(
    followup_prompts: list[FollowupPrompt], modifier: str
) -> FollowupPrompt:
    return next(filter(lambda x: x.modifier == modifier, followup_prompts))


def load_mturk_data(dir: str) -> pd.DataFrame:
    files = os.listdir(dir)
    df = pd.concat(
        [
            pd.read_csv(os.path.join(dir, file))
            for file in files
            if file.endswith(".csv")
        ],
        ignore_index=True,
    )

    df = (
        df[["WorkerId", "Input.mod_eval", "Input.mod_indep", "Answer.thesaurus.label"]]
        .groupby(["Input.mod_eval", "Input.mod_indep"])["Answer.thesaurus.label"]
        .agg(unique_count=lambda x: x.nunique(), mode=lambda x: x.mode()[0])
    )

    return df  # want a majority


def compute_similarity_pairs(
    grads,
    similarity_threshold: float,
    followup_prompts: list[FollowupPrompt],
    negate: bool = False,
) -> list[tuple[str, str, float]]:
    mean_grad = {}
    modifiers = list(sorted(grads.keys()))
    for mod in modifiers:
        grad_list = [each for each in grads[mod] if each is not None]
        mean_grad[mod] = np.stack(grad_list).mean(axis=0)

    result_list = []

    for mod_1 in modifiers:
        for mod_2 in modifiers:
            if mod_1 == mod_2 or mod_1 == "" or mod_2 == "":
                continue

            mod_eval_prompt = get_followup_prompt(followup_prompts, mod_1)
            mod_indep_prompt = get_followup_prompt(followup_prompts, mod_2)
            if (
                mod_indep_prompt.edit_pair is not True
                or mod_eval_prompt.edit_pair is True
                or mod_eval_prompt.exclude_eval
            ):
                # print("Skipping", mod_1, mod_2)
                continue
            # pairs where this is true will be included twice (symmetric)
            cs = cosine_sim(mean_grad[mod_1][0], mean_grad[mod_2][0])
            if (not negate and cs > similarity_threshold) or (
                negate and cs < similarity_threshold
            ):
                result_list.append((mod_1, mod_2, float(cs)))

    return result_list


def main():
    args = parse_args()

    assert (
        args.followup_prompts in args.grad_dir
    ), f"Expected {args.followup_prompts} in {args.grad_dir}"
    assert args.model in args.grad_dir, f"Expected {args.model} in {args.grad_dir}"
    followup_prompts = FOLLOWUP_PROMPT_DATASET[args.followup_prompts]

    with open(os.path.join(args.grad_dir, "grads.pkl"), "rb") as f:
        grads = pickle.load(f)
    similarity_pairs = compute_similarity_pairs(
        grads, args.threshold, followup_prompts, args.negate
    )
    random.shuffle(similarity_pairs)

    mturk_df = load_mturk_data(args.mturk_dir)
    if args.negate:
        operational_thesaurus = set(
            mturk_df[
                (mturk_df["mode"] == "Expected") & (mturk_df["unique_count"] <= 1)
            ].index.values.tolist()
        )
    else:
        operational_thesaurus = set(
            mturk_df[
                (mturk_df["mode"] == "Unexpected") & (mturk_df["unique_count"] <= 1)
            ].index.values.tolist()
        )

    all_labeled = set(mturk_df.index.values.tolist())

    print("unexpecteds", len(operational_thesaurus), "labeled", len(all_labeled))

    matching_pairs = list(
        filter(lambda x: (x[0], x[1]) in operational_thesaurus, similarity_pairs)
    )
    not_matching_pairs = list(
        filter(lambda x: (x[0], x[1]) not in operational_thesaurus, similarity_pairs)
    )

    unlabeled = list(
        filter(lambda x: (x[0], x[1]) not in all_labeled, similarity_pairs)
    )

    print("Found", len(matching_pairs), "matching pairs")
    print("Found", len(not_matching_pairs), "not matching pairs")
    print("Found", len(unlabeled), "unlabeled pairs of", len(similarity_pairs), "pairs")
    print(json.dumps(matching_pairs))

    # avg cosine similarity of matching pairs
    print(
        "Average cosine similarity of matching pairs:",
        np.mean([x[2] for x in matching_pairs]),
    )

    # avg cosine similarity of not matching pairs
    print(
        "Average cosine similarity of not matching pairs:",
        np.mean([x[2] for x in not_matching_pairs]),
    )

    # get month and day like "sep30"
    today = datetime.date.today()

    matches_filename = os.path.join(
        args.grad_dir,
        f"mturk_{today.strftime('%b%d')}_{'negate' if args.negate else 'pos'}_{args.threshold}.json",
    )
    print("Saving to", matches_filename)
    with open(matches_filename, "w") as f:
        json.dump(matching_pairs[:30], f)

    # to_label_filename = os.path.join(
    #     "to_label3",
    #     f"{args.followup_prompts}_{args.model}_{'negate' if args.negate else 'pos'}_{args.threshold}.json",
    # )
    # print('Saving to', to_label_filename)
    # os.makedirs(os.path.dirname(to_label_filename), exist_ok=True)
    # with open(to_label_filename, "w") as f:
    #     json.dump(unlabeled, f)


if __name__ == "__main__":
    main()
