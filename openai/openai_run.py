# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import json
import argparse
import logging

from openai import OpenAI
from openai_prompt import (
    batch_prompt_direct_input,
    batch_prompt_cot_input,
    batch_prompt_direct_output,
    batch_prompt_cot_output,
)

logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="input",
        choices=["input", "output"],
        help="Whether to prompt with input or output",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Whether to use chain-of-thought prompting (only applicable for input mode)",
    )
    parser.add_argument(
        "--model",
        "-mo",
        type=str,
        default="gpt-3.5-turbo-0613",
        help="Model name or path",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for hosted vLLM API (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--generation-args",
        type=json.loads,
        default="{}",
        help='Additional generation arguments as a JSON string (e.g., \'{"temperature": 0.7, "max_tokens": 100}\')',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/openai",
        help="Directory to save generated outputs",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Completions to sample per problem. pass@5 needs at least 5 AND a "
        "non-zero temperature -- with greedy decoding every completion is "
        "identical, so pass@5 collapses to pass@1 at n times the cost. Use 1 if "
        "only pass@1 is wanted (default: 10).",
    )
    args = parser.parse_args()
    return args


def run_openai(args):
    dataset = [
        json.loads(line) for line in open("../data/cruxeval.jsonl", "r").readlines()
    ]

    max_samples = args.generation_args.get("max_samples", -1)
    if max_samples > 0:
        logger.info(f"Limiting to max_samples={max_samples}")
        dataset = dataset[:max_samples]

    if args.mode == "input":
        prompts = [(data["code"], data["output"]) for data in dataset]
    else:
        prompts = [(data["code"], data["input"]) for data in dataset]

    if args.cot:
        logger.debug("Using chain-of-thought prompting, setting max_tokens to 1000")
        args.generation_args["max_tokens"] = 1000
    else:
        logger.debug("Not using chain-of-thought prompting, setting max_tokens to 100")
        args.generation_args["max_tokens"] = 100

    client = OpenAI(base_url=args.url, api_key=os.getenv("API_KEY"))

    fn = {
        (True, "input"): batch_prompt_cot_input,
        (True, "output"): batch_prompt_cot_output,
        (False, "input"): batch_prompt_direct_input,
        (False, "output"): batch_prompt_direct_output,
    }[(args.cot, args.mode)]

    if args.n_samples < 5:
        logger.warning(
            f"n_samples={args.n_samples} < 5: pass@5 is not estimable and will be "
            "reported as if every problem had only that many attempts."
        )
    if not args.generation_args.get("temperature"):
        logger.warning(
            "temperature is 0/unset: all %d completions per problem will be "
            "identical, so pass@5 will equal pass@1 at %dx the cost.",
            args.n_samples,
            args.n_samples,
        )

    outputs = fn(
        client,
        prompts,
        n=args.n_samples,
        model=args.model,
        stop=["[/ANSWER]"],
        **args.generation_args,
    )
    save_dir = os.path.join(args.output_dir, "generations.json")
    outputs_dict = {f"sample_{i}": [j[0] for j in o] for i, o in enumerate(outputs)}
    json.dump(outputs_dict, open(save_dir, "w"))
    return outputs


if __name__ == "__main__":
    args = parse_args()
    run_openai(args)
