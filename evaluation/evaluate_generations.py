# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from utils_general import (
    evaluate_score,
    pass_at_k,
)


def evaluate_generations(generations: dict[str, list], mode):
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

    # Bound the pool. Every generation is scored by check_correctness(), which
    # forks its OWN multiprocessing.Process and returns False if it cannot finish
    # within timeout+1 seconds. An unbounded ProcessPoolExecutor sizes itself to
    # os.cpu_count() -- the NODE's core count (128 on a Leonardo boost node), not
    # this job's cgroup allocation -- so it launches ~cpu_count workers each
    # forking a child, and process startup alone can exceed the 4s budget. The
    # failures that follow are indistinguishable from wrong answers, and they are
    # load-dependent, not deterministic: the same 800 domynedge-sft-37410
    # CRUXEval-I generations scored 72.74 in the original run, 51.93 under an
    # unbounded pool, and 76.11 scored serially. Capping is what makes the number
    # reproducible.
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
    pass_at_1s, pass_at_5s = [], []
    for execution_result in all_scores:
        c, n_completions = execution_result.count(True), len(execution_result)
        pass_at_1s.append(pass_at_k(n_completions, c, 1))
        pass_at_5s.append(pass_at_k(n_completions, c, 5))

    return {
        "raw_generations": generations,
        "raw_scored_generations": {
            f"sample_{i}": all_scores[i] for i in range(num_problems)
        },
        "pass_at_1": sum(pass_at_1s) / len(pass_at_1s) * 100,
        "pass_at_5": sum(pass_at_5s) / len(pass_at_5s) * 100,
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

    args = parser.parse_args()
    generations = json.load(open(args.generations_path, "r"))
    print(f"Scoring {args.generations_path}... expect around a minute")

    # An explicit --mode wins. The flag was defined but never read: the mode was
    # sniffed from the substring "input" in the FILE PATH, so scoring input-mode
    # generations from any path without that word silently graded them as output
    # mode. It cost 21 points of pass@1 on a re-score run out of a temp directory
    # and looked exactly like a regression in the answer extractor. Path sniffing
    # is kept only as the fallback, for callers that pass no --mode.
    if args.mode not in ("input", "output"):
        args.mode = "input" if "input" in args.generations_path else "output"

    results = evaluate_generations(generations, args.mode)
    print("Finished!")
    print(
        "pass@1:",
        round(results["pass_at_1"], 1),
        "pass@5:",
        round(results["pass_at_5"], 1),
    )
    if args.scored_results_path is not None:
        print(f"Dumping to {args.scored_results_path}")
        json.dump(results, open(args.scored_results_path, "w"))
