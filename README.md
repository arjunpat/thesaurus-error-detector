
# TED

## Requirements

You will need at least 1 Nvidia GPU with approximately 40 GB of VRAM to successfully run the majority of this code. Running Meta Llama 70B Instruct will require about ~200 GB combined VRAM. However, this can be avoided by computing the reference thesaurus using GPT-4 or another API.

## Installation

Create the environment.

```bash
conda create -n ted python=3.11
pip install -r requirements.txt

conda activate ted
```

## Running TED

### Construct Operational Embeddings

#### Generate Data

First, compute responses and edits to ethics questions (ethics2 and ethics3 sets).

```bash
python create_user_query_dataset.py --modal mistral --followup constitution7 --dataset ethics2
python create_user_query_dataset.py --modal mistral --followup constitution7 --dataset ethics3
```

We will use ethics2 for test and ethics3 for computing operational embeddings.

#### Compute Gradients

We can now compute gradients with respect to the modified outputs to construct our operational embeddings.

```bash
python compute_grads.py --model mistral --dir <ethics3 dataset folder>
```

This will compute grads and put them into the same folder in a file named `grads.pkl`. We can now compute potential failures.

#### Construct Thesauruses and Find Disagreements

```bash
python compute_differences.py --model llama3-70B --grad-dir <ethics3 dataset folder> --followup-prompts constitution7 --similarity-threshold 0.95
```

This code takes care of generating the reference thesaurus on demand using the model specified by `--model`. It also genreates the LLM thesaurus based on `--similarity-threshold`, which is the same as Tau in the paper.

It can also take precomputed references by using the `--similarity-file` and `--intersection-file` arguments. These precomputed references can be made with GPT-4 by running `create_similarity_pairs.py`, for example.

The above command will produce candidate unexpected side-effect failures and save them into a JSON file in the same directory. To produce candidate inadequate update failures, add the `--negate` flag, and set `--similarity-threshold` to -0.5 as a good starting point.

#### Grade Pairs on Downstream Success

```bash
python grade_outputs_parallel.py --modifiers <JSON file of candidates> --dataset <ethics2 outputs pickle file> --model gpt4 --followup-prompts constitution7
```

`notebooks/eval_queries_mistral2.ipynb` contains helpful code for tallying the results of grading.

#### Inference-steering failures

For inference-steering, we want to use the `summarize2` followup prompts, instead of `constitution7`. We can generate the necessary data:

```bash
python create_user_query_dataset.py --modal mistral --followup summarize2 --dataset ethics3
python create_inference_responses_dataset.py --model mistral --infer-prompts infer1 --followup-promots summarize2
```

Compute gradients.

```bash
python compute_grads.py --model mistral --dir <ethics3, summarize2 dataset folder>
```


Find candidates.

```bash
python compute_differences.py --model llama3-70B --grad-dir <ethics3, summarize2 dataset folder> --followup-prompts summarize2 --similarity-threshold 0.95
```

Grade inference-steering candidates.

```bash
python grade_outputs_parallel.py --modifiers <JSON file of candidates> --dataset <inference outputs pickle file> --model gpt4 --followup-prompts summarize2
```
