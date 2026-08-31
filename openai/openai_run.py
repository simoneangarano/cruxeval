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

    # CoT needs room for a written-out trace before [ANSWER]; a direct answer needs
    # almost none. These stay as fallbacks for a standalone run, but they no longer
    # overwrite a configured budget: a thinking model spends thousands of tokens in
    # reasoning_content before it writes any answer, so the hardcoded 1000 cut every
    # generation off mid-thought and returned empty content for all of them.
    default_max_tokens = 1000 if args.cot else 100
    if not args.generation_args.get("max_tokens"):
        args.generation_args["max_tokens"] = default_max_tokens
    logger.info("max_tokens=%d", args.generation_args["max_tokens"])

    # The stop sequence is how upstream cuts the generation at the end of the
    # answer, but a reasoning model reasons *about the required format* and quotes
    # "[/ANSWER]" while doing so -- generation then stops inside the trace, before
    # any answer is written (observed: finish_reason="stop", content empty). The
    # extraction functions cut the closing tag themselves now, so dropping the stop
    # here costs only a few trailing tokens.
    reasoning = bool(args.generation_args.get("reasoning"))
    stop = None if reasoning else ["[/ANSWER]"]
    if reasoning:
        logger.info(
            "reasoning=true: not passing a stop sequence (a thinking model can trip "
            "[/ANSWER] mid-trace and never reach the answer)"
        )

    # The 600s SDK default is not enough for a thinking model on a busy swarm: the
    # run died on APITimeoutError with zero generations written.
    client = OpenAI(
        base_url=args.url,
        api_key=os.getenv("API_KEY"),
        timeout=float(os.getenv("CRUXEVAL_TIMEOUT", "1800")),
        max_retries=int(os.getenv("CRUXEVAL_MAX_RETRIES", "5")),
    )

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
        stop=stop,
        **args.generation_args,
    )
    save_dir = os.path.join(args.output_dir, "generations.json")
    outputs_dict = {f"sample_{i}": [j[0] for j in o] for i, o in enumerate(outputs)}
    json.dump(outputs_dict, open(save_dir, "w"))
    return outputs


if __name__ == "__main__":
    # Without a handler, logging's lastResort only emits WARNING and above with no
    # context, so every logger.info here (the max_samples limit, the effective
    # token budget, the pass@5 caveats) was silently dropped and a failure showed
    # up as a bare traceback in the driver's log.
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    args = parse_args()
    run_openai(args)
