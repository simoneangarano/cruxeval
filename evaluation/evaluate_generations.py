# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from utils_general import (
    evaluate_score,
    pass_at_k,
)


def evaluate_generations(generations: dict[str, list], mode: str, k_values: list[int]):
    # Load the samples
    dataset = [
        json.loads(line) for line in open("../data/cruxeval.jsonl", "r").readlines()
    ]
    references = [(doc["code"], doc["input"], doc["output"]) for doc in dataset]

    # Score only as many samples as were generated (generation may be capped via
    # max_samples); align references to the generated subset.
    num_problems = len(generations)
    references = references[:num_problems]

    # Run the samples
    try:
        generations_list = [generations[f"sample_{i}"] for i in range(num_problems)]
    except KeyError:
        assert False, (
            "check format of generations, should be dictionary of lists with keys of id's in the form sample_i"
        )

    # An unbounded pool sizes itself to os.cpu_count() -- the NODE's cores, not the
    # job's cgroup -- and each worker forks a child inside check_correctness()'s
    # few-second budget, so correct answers time out and score as wrong. The effect
    # is load-dependent: identical generations scored 72.74 / 51.93 / 76.11.
    max_workers = int(os.environ.get("CRUXEVAL_SCORE_WORKERS", "8"))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        args_list = zip(generations_list, references, [mode] * len(generations_list))
        results = executor.map(evaluate_score, args_list)
    all_scores = list(results)

    # Compute pass@k scores. `n_completions` must not be called `n`: it is
    # per-problem (the number of generations sampled for one problem), and reusing
    # the problem-count name here previously leaked the loop value into the
    # raw_scored_generations comprehension below, truncating it to the number of
    # completions (10) instead of the number of problems.
    pass_at = defaultdict(list)
    for execution_result in all_scores:
        c, n_completions = execution_result.count(True), len(execution_result)
        for k in k_values:
            pass_at[f"pass_at_{k}"].append(pass_at_k(n_completions, c, k))

    return {
        "raw_generations": generations,
        "raw_scored_generations": {
            f"sample_{i}": all_scores[i] for i in range(num_problems)
        },
        **{f"pass_at_{k}": sum(pass_at[f"pass_at_{k}"]) / len(pass_at[f"pass_at_{k}"]) * 100 for k in k_values},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations_path",
        help="JSON path containing outputs to evaluate. Should contain a list of \
              length 800, where each element is a list of different generations \
              for that benchmark sample.",
        type=str,
    )
    parser.add_argument(
        "--scored_results_path",
        help="path to dump scored results",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mode",
        help="either input or output, depending on which one to evaluate",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--k",
        help="k for pass@k evaluation",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    generations = json.load(open(args.generations_path, "r"))
    print(f"Scoring {args.generations_path}... expect around a minute")

    # The flag was defined but never read; the mode was sniffed from the file path,
    # so input-mode generations under any other path were graded as output mode
    # (worth 21 points of pass@1). Path sniffing stays as the fallback.
    if args.mode not in ("input", "output"):
        args.mode = "input" if "input" in args.generations_path else "output"

    k_values = []
    args.k = max(args.k, 1)  # Ensure k is at least 1
    power = 1
    while power < args.k:
        k_values.append(power)
        power *= 2
    if args.k not in k_values:
        k_values.append(args.k)

    results = evaluate_generations(generations, args.mode, k_values)
    print("Finished!")
    for k in k_values:
        print(
            f"pass@{k}:",
            round(results[f"pass_at_{k}"], 1),
        )
    if args.scored_results_path is not None:
        print(f"Dumping to {args.scored_results_path}")
        json.dump(results, open(args.scored_results_path, "w"))
