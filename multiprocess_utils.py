import os

import ray
import torch
import torch.multiprocessing as mp

from utils.llm import LLAMA3, ChatLLM, Mistral


def load_chatllm(model: str) -> ChatLLM:
    if model == "llama3":
        llm = LLAMA3("Meta-Llama-3-8B-Instruct", gpus=1)
    elif model == "llama3-70B":
        ray.shutdown()
        ray.init(num_gpus=torch.cuda.device_count())

        llm = LLAMA3("Meta-Llama-3-70B-Instruct", gpus=torch.cuda.device_count())
    elif model == "mistral":
        llm = Mistral("mistral-7B-instruct-v0.2", gpus=1)
    else:
        raise ValueError("Invalid model: " + model)

    return llm


def gen_process(
    model: str,
    prompts: list[str],
    max_tokens: int,
    results_queue: mp.Queue,
    rank: int,
) -> list[str]:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
    outputs = load_chatllm(model).generate_batch(
        prompts, max_tokens=max_tokens, use_tqdm=True
    )
    results_queue.put((rank, outputs))


def gen_batch(
    model: str, first_model: ChatLLM, prompts: list[str], max_tokens: int | None = None
) -> list[str]:
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0

    n_per_gpu = len(prompts) // num_gpus
    assert n_per_gpu > 0

    processes = []
    results = mp.Queue(num_gpus - 1)
    for rank in range(1, num_gpus):
        start = rank * n_per_gpu
        end = (rank + 1) * n_per_gpu
        if rank == num_gpus - 1:
            end = len(prompts)

        print(f"Starting {rank} of {num_gpus - 1}")
        p = mp.Process(
            target=gen_process,
            args=(model, prompts[start:end], max_tokens, results, rank),
        )
        p.start()
        processes.append(p)

    # can't send loaded llm to new process
    first_outputs = first_model.generate_batch(
        prompts[:n_per_gpu], max_tokens=max_tokens, use_tqdm=True
    )

    print("Waiting for processes to finish")
    outputs = [(0, first_outputs)]
    for _ in range(len(processes)):
        outputs.append(results.get())

    for p in processes:
        p.join()

    outputs.sort(key=lambda x: x[0])
    # flatten
    outputs = [x for xs in map(lambda x: x[1], outputs) for x in xs]

    return outputs
