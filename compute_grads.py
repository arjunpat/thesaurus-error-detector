import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from utils import get_datetime_str
from utils.transformer import load_llama, load_llama3, load_mistral


def get_raw_embedding_table(model):
    return model.get_input_embeddings()._parameters["weight"]


def cosine_similarity(a: np.ndarray, b: np.ndarray, normalize=True) -> float:
    if not normalize:
        return np.dot(a, b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def consolidate_by_query(results: list[dict]) -> dict:
    result_dict = {}

    for result in results:
        user_query = result["user_query"]
        modifier = result["modifier"]

        if user_query not in result_dict:
            result_dict[user_query] = {}

        result_dict[user_query][modifier] = result

    return result_dict


def compute_gradient_for_tokens(
    model,
    tokenizer,
    embedding_table,
    prompt: str,
    expected_output: str,
    prompt_idx_to_compute_grad: int,
    toks_to_save: int,
):
    """
    toks_to_save is the number of tokens to save after the prompt_idx_to_compute_grad
    """
    assert len(prompt) > 0 and prompt_idx_to_compute_grad < len(
        prompt
    ), "Invalid prompt"
    # assert prompt[-1] != " " and (
    #     expected_output[0] == " " or expected_output[0] == "\n"
    # ), "prompt should not end with space, and expected_output should start with space"

    before_prompt = prompt[:prompt_idx_to_compute_grad]

    n_before_prompt_toks = len(
        tokenizer(before_prompt, return_tensors="pt").input_ids[0]
    )
    prompt_toks = tokenizer(prompt, return_tensors="pt").input_ids[0]
    toks = tokenizer(prompt + expected_output, return_tensors="pt").input_ids[0]
    assert len(toks.shape) == 1, "toks shape is " + str(toks.shape)
    output_toks = toks[len(prompt_toks) :]

    # embed the toks
    full_embeddings = embedding_table[toks].unsqueeze(0).detach()

    # setup gradient computations
    if full_embeddings.requires_grad:
        full_embeddings.grad.zero_()
    full_embeddings.requires_grad = True
    full_embeddings.retain_grad()

    # create labels of -100 for the prompt and tok id for the output
    prompt_labels = torch.tensor([-100] * len(prompt_toks))
    labels = torch.cat([prompt_labels, output_toks.squeeze(0)]).long().unsqueeze(0)
    assert labels.shape == (1, len(toks)), "labels shape is " + str(labels.shape)

    # print("Prompt: ", tokenizer.decode(prompt_toks))
    # print("Output: ", tokenizer.decode(output_toks))

    # forward + backward pass of model
    out = model(inputs_embeds=full_embeddings, labels=labels)
    out.loss.backward()

    assert len(full_embeddings.grad[0]) == len(toks), "length mismatch"

    ret_val = (
        full_embeddings.grad[0][
            n_before_prompt_toks : n_before_prompt_toks + toks_to_save
        ]
        .detach()
        .cpu()
        .numpy()
    )

    del prompt_labels
    del labels
    del prompt_toks
    del toks
    del full_embeddings
    del out
    model.zero_grad()
    torch.cuda.empty_cache()

    return ret_val


def test():
    model, tokenizer = load_llama3("Meta-Llama-3-8B-Instruct")
    embedding_table = get_raw_embedding_table(model)

    # space must preceed the output because of the tokenizer
    prompt = "Hello there, how are"
    output = " superfragilistic alidocious"

    grad = compute_gradient_for_tokens(
        model,
        tokenizer,
        embedding_table,
        prompt,
        output,
        prompt_idx_to_compute_grad=0,
        toks_to_save=5,
    )
    print(grad.shape)
    print(grad)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--pair-modifiers", type=str, required=False, default=None)
    return parser.parse_args()


def load_model(model_name: str, device):
    if model_name == "llama":
        model, tokenizer = load_llama("llama-7B-chat", device=device)
        raise "do not use"
    elif model_name == "mistral":
        model, tokenizer = load_mistral("mistral-7B-instruct-v0.2", device=device)
    elif model_name == "llama3":
        model, tokenizer = load_llama3("Meta-Llama-3-8B-Instruct", device=device)
    else:
        raise NotImplementedError
    embedding_table = get_raw_embedding_table(model)
    return model, tokenizer, embedding_table


def process_subset(
    rank: int, dataset_subset, args, queue: mp.Queue, control_mod: str = ""
):
    print(
        "Starting rank " + str(rank) + " with " + str(len(dataset_subset)) + " queries"
    )
    device = torch.device(f"cuda:{rank}")

    adjective_to_grads = {}
    model, tokenizer, embedding_table = load_model(args.model, device)

    assert model.device == device, "model device is " + str(model.device)

    for user_query in tqdm(dataset_subset, desc=f"Rank {rank}"):
        for modifier in dataset_subset[user_query]:
            if modifier not in adjective_to_grads:
                adjective_to_grads[modifier] = []
            prompt = dataset_subset[user_query][control_mod]["query"]  # control
            query_start = dataset_subset[user_query][control_mod]["query_start"]

            # 10000 limit is usually good, otherwise CUDA out of memory
            output = dataset_subset[user_query][modifier]["output"][:10000]

            try:
                grad = compute_gradient_for_tokens(
                    model,
                    tokenizer,
                    embedding_table,
                    prompt,
                    output,
                    prompt_idx_to_compute_grad=query_start,
                    toks_to_save=4,
                )
                adjective_to_grads[modifier].append(grad)
            except Exception as e:
                adjective_to_grads[modifier].append(None)  # maintain ordering
                print(f"Error computing grads for {user_query} {modifier}")
                print(e)

    print("Finished rank " + str(rank))
    queue.put((rank, adjective_to_grads))


def run_grads(
    dataset, control_mod: str, modifier: str, model, tokenizer, embedding_table
):
    grads = []
    for user_query in tqdm(dataset, desc=f"{control_mod} {modifier}"):
        prompt = dataset[user_query][control_mod]["query"]  # control
        query_start = dataset[user_query][control_mod]["query_start"]

        # 10000 limit is usually good, otherwise CUDA out of memory
        output = dataset[user_query][modifier]["output"][:10000]

        try:
            grad = compute_gradient_for_tokens(
                model,
                tokenizer,
                embedding_table,
                prompt,
                output,
                prompt_idx_to_compute_grad=query_start,
                toks_to_save=4,
            )
            grads.append(grad)
        except Exception as e:
            grads.append(None)  # maintain ordering
            print(
                f"Error computing grads for {user_query} {modifier} with {control_mod}"
            )
            print(e)

    return grads


def process_pairs(
    rank: int, args, dataset, pairs: list[list[str, str]], queue: mp.Queue
):
    print("Starting rank " + str(rank) + " with " + str(len(pairs)) + " pairs")

    device = torch.device(f"cuda:{rank}")
    model, tokenizer, embedding_table = load_model(args.model, device)
    assert model.device == device, "model device is " + str(model.device)

    computed_grads = {}

    for pair_idx in range(len(pairs)):
        print(f"Rank {rank}: Pair {pair_idx + 1} ({pairs[pair_idx]}) of {len(pairs)}")

        control_mod = pairs[pair_idx][0]
        modifier = pairs[pair_idx][1]

        computed_grads[(control_mod, modifier)] = run_grads(
            dataset, control_mod, modifier, model, tokenizer, embedding_table
        )

        control_mod = pairs[pair_idx][1]
        modifier = pairs[pair_idx][0]

        computed_grads[(control_mod, modifier)] = run_grads(
            dataset, control_mod, modifier, model, tokenizer, embedding_table
        )

    print("Finished rank " + str(rank))
    queue.put(computed_grads)


def compare_pairs(args, dataset):
    with open(args.pair_modifiers, "r") as f:
        pairs = json.load(f)  # [[a, b], [a, b], ...]

    num_gpus = torch.cuda.device_count()
    subset_size = len(pairs) // num_gpus

    mp.set_start_method("spawn")
    processes = []
    results = mp.Queue(num_gpus)
    for rank in range(num_gpus):
        start_idx = rank * subset_size
        end_idx = (rank + 1) * subset_size if rank < num_gpus - 1 else len(pairs)
        pair_subset = pairs[start_idx:end_idx]

        if len(pair_subset) > 0:
            p = mp.Process(
                target=process_pairs, args=(rank, args, dataset, pair_subset, results)
            )
            p.start()
            processes.append(p)

    print("Waiting for processes to finish")
    computed_grads = {}

    for _ in range(num_gpus):
        computed_grads.update(results.get())

    for p in processes:
        p.join()

    file_name = os.path.join(args.dir, f"grad_pairs_{get_datetime_str()}.pkl")
    print("Saving", file_name)

    with open(file_name, "wb") as f:
        pickle.dump(
            {
                "grads": computed_grads,
                "pair_modifiers": args.pair_modifiers,
                "pairs": pairs,
            },
            f,
        )


def main():
    args = parse_args()
    dataset_file = os.path.join(args.dir, "generated.pkl")
    with open(dataset_file, "rb") as f:
        dataset = consolidate_by_query(pickle.load(f))

    if args.pair_modifiers is not None:
        return compare_pairs(args, dataset)

    num_gpus = torch.cuda.device_count()
    dataset_keys = list(sorted(dataset.keys()))
    subset_size = len(dataset_keys) // num_gpus

    mp.set_start_method("spawn")
    processes = []
    results = mp.Queue(num_gpus)
    for rank in range(num_gpus):
        start_idx = rank * subset_size
        end_idx = (rank + 1) * subset_size if rank < num_gpus - 1 else len(dataset_keys)
        dataset_subset = {key: dataset[key] for key in dataset_keys[start_idx:end_idx]}

        p = mp.Process(
            target=process_subset, args=(rank, dataset_subset, args, results)
        )
        p.start()
        processes.append(p)

    print("Waiting for processes to finish")

    grads = []
    for _ in range(num_gpus):
        grads.append(results.get())

    print("All processes finished")
    for p in processes:
        p.join()

    # ensure that grads are in the same order as the dataset keys
    adjective_to_grads = {}
    for rank, subset_grads in sorted(grads, key=lambda x: x[0]):
        for modifier, grads in subset_grads.items():
            if modifier not in adjective_to_grads:
                adjective_to_grads[modifier] = []
            adjective_to_grads[modifier].extend(grads)

    print("Saving", args.dir)

    with open(os.path.join(args.dir, "grads.pkl"), "wb") as f:
        pickle.dump(adjective_to_grads, f)


if __name__ == "__main__":
    main()
