#!
import json
import logging
import os
import os as _os
import re
import sys
import sys as _sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

_sys.path.append(
    _os.environ.get("ROOT") or _os.path.abspath(__file__).split("/suites/")[0]
)
import token_budget

sys.path.append("..")
from prompts import (
    make_cot_input_prompt,
    make_cot_output_prompt,
    make_direct_input_prompt,
    make_direct_output_prompt,
)

logger = logging.getLogger(__name__)

_empty_completions = Counter()
_empty_lock = threading.Lock()


def _strip_reasoning(gen) -> str:
    """Drop the reasoning trace, which is where spurious [ANSWER] tags live."""
    gen = gen or ""
    if "</think>" in gen:
        return gen.rsplit("</think>", 1)[1]
    if "<think>" in gen:
        return gen.rsplit("<think>", 1)[1]
    return gen


def _last_answer_block(gen):
    """Text after the LAST [ANSWER] tag, or None if there is none."""
    if "[ANSWER]" not in gen:
        return None
    return gen.rsplit("[ANSWER]", 1)[1].strip()


_FENCE_RE = re.compile(r"```(?:\w+)?[ \t]*\r?\n(.*?)```", re.DOTALL)
# A bare `f(...) == value` line, for models that drop the `assert` keyword.
_BARE_CALL_RE = re.compile(r"^f\s*\(.*\)\s*==")


def _fenced_assert(gen):
    """Last `assert f(...) == ...` line inside a fenced block, or None."""
    for block in reversed(_FENCE_RE.findall(gen)):
        for line in reversed(block.splitlines()):
            line = line.strip()
            if "==" not in line or "f(" not in line:
                continue
            if line.startswith("assert ") or _BARE_CALL_RE.match(line):
                return line
    return None


def _answer_body(gen) -> str:
    """The generation with the closing tag (and anything after it) removed."""
    return (gen or "").split("[/ANSWER]")[0]


def extract_answer_direct_output(gen):
    gen = _answer_body(gen)
    if "==" in gen:
        gen = gen.split("==")[1]
    return gen.strip()


def extract_answer_direct_input(gen):
    gen = _answer_body(gen)
    if "==" in gen:
        gen = gen.split("==")[0].strip()
    if "assert f" in gen:
        gen = "f" + gen.split("assert f")[1].strip()
    return gen.strip()


def extract_answer_cot_input(gen):
    gen = _answer_body(_strip_reasoning(gen))
    block = _last_answer_block(gen)
    if block is None:
        block = _fenced_assert(gen)
    if block is not None:
        if "==" in block:
            block = block.split("==")[0]
        if "assert f" in block:
            block = "f" + block.split("assert f")[1].strip()
        return block.strip()
    return gen.split("\n")[-1].strip()


def extract_answer_cot_output(gen):
    gen = _answer_body(_strip_reasoning(gen))
    block = _last_answer_block(gen)
    if block is None:
        block = _fenced_assert(gen)
    if block is not None:
        if "==" in block:
            block = block.split("==")[1]
        return block.strip()
    return gen.split("\n")[-1].strip()


NON_INFERENCE_ARGS = frozenset(
    {
        "max_samples",
        "num_threads",
        "reasoning",
        "system_prompt_template",
        "tokenizer_path",
        "think",
        "nothink",
    }
)

# Sent as explicit arguments below, so they must not be repeated in extra_body.
EXPLICIT_ARGS = frozenset({"temperature", "max_tokens", "top_p", "presence_penalty"})


def sampling_params(generation_args: dict) -> dict:
    """Inference params only, with the explicitly-passed ones removed.

    What is left is the vLLM-specific extras (min_p, top_k, repetition_penalty,
    chat_template_kwargs, ...) that belong in extra_body.
    """
    return {
        k: v
        for k, v in generation_args.items()
        if k not in NON_INFERENCE_ARGS and k not in EXPLICIT_ARGS and v is not None
    }


def make_cache_key(full_prompt: str, model: str, generation_args: dict) -> str:
    """Cache key covering everything that changes the generations.

    The key previously covered only the prompt, model and (when non-zero)
    temperature, so re-running the same model at the same temperature with a
    different top_p / top_k / min_p / penalty silently replayed the old
    generations. Every inference param is included now, and a temperature of 0 is
    no longer a special case that drops the rest.
    """
    params = sampling_params(generation_args)
    for key in ("temperature", "max_tokens", "top_p", "presence_penalty"):
        if generation_args.get(key) is not None:
            params[key] = generation_args[key]
    fingerprint = json.dumps(params, sort_keys=True, default=str)
    return f"{full_prompt}_{model}_{fingerprint}"


def call_openai_api(
    client, system_prompt, prompt, n, model, stop, **generation_args
) -> list[str]:
    # print("not cached")
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    result = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=generation_args["temperature"],
        n=n,
        # Keep prompt + completion inside the context window; vLLM 400s otherwise.
        max_tokens=token_budget.resolve_max_tokens(generation_args, messages=prompt),
        stop=stop,
        top_p=generation_args["top_p"],
        presence_penalty=generation_args["presence_penalty"],
        extra_body=sampling_params(generation_args),
    )
    # print(f"OpenAI API response: {result}")
    return [_completion_text(result.choices[i], model) for i in range(n)]


def _completion_text(choice, model: str) -> str:
    """The answer text of one choice, or "" when the model produced none.

    A reasoning model can return `message.content` empty: it either spent the
    whole budget inside reasoning_content (finish_reason="length") or tripped a
    stop sequence while still thinking. That None used to reach the extraction
    functions and take the entire run down with a TypeError -- after every
    generation had already been paid for. Score it as a miss instead, and say
    loudly why, since a run full of these scores ~0 for reasons that have nothing
    to do with the model's code reasoning.
    """
    content = choice.message.content
    if content:
        return content

    # vLLM names this field `reasoning` or `reasoning_content` depending on version.
    reasoning = (
        getattr(choice.message, "reasoning", None)
        or getattr(choice.message, "reasoning_content", None)
        or ""
    )
    with _empty_lock:
        _empty_completions[choice.finish_reason or "unknown"] += 1
        first = sum(_empty_completions.values()) == 1
    if first:  # one detailed warning; batch_prompt reports the final tally
        logger.warning(
            "%s returned no answer (finish_reason=%s, %d chars of reasoning): "
            "scoring it as a miss. %s",
            model,
            choice.finish_reason,
            len(reasoning),
            (
                "The budget ran out mid-reasoning -- raise generation.max_tokens."
                if choice.finish_reason == "length"
                else "Generation stopped before the answer was written."
            ),
        )
    return ""


def prompt_openai_general(
    client,
    make_prompt_fn,
    i,
    cache,
    gpt_query,
    n,
    model,
    stop,
    **generation_args,
) -> tuple[str, list[str]]:
    full_prompt = make_prompt_fn(gpt_query)
    cache_key = make_cache_key(full_prompt, model, generation_args)

    cached = [gen for gen in cache.get(cache_key, []) if gen is not None]

    if len(cached) < n:
        system_prompt = "You are an expert at Python programming, code execution, test case generation, and fuzzing."
        result = cached + call_openai_api(
            client,
            system_prompt,
            full_prompt,
            n=n - len(cached),
            model=model,
            stop=stop,
            **generation_args,
        )
        cache[cache_key] = result
    else:
        result = cached[:n]
    return i, (cache_key, result)  # type: ignore


def batch_prompt(fn, extraction_fn, client, queries, n, model, stop, **generation_args):
    # load the cache
    CACHE_DIR_PREFIX = ""
    cache_dir = os.path.join(CACHE_DIR_PREFIX, "cache.json")
    cache_dir_tmp = os.path.join(CACHE_DIR_PREFIX, "cache.json.tmp")
    cache_dir_bak = os.path.join(CACHE_DIR_PREFIX, "cache.json.bak")
    try:
        with open(cache_dir, "r") as f:
            cache = json.load(f)
    except Exception:
        with open(cache_dir, "w") as f:
            json.dump({}, f)
        cache = {}

    max_workers = int(os.getenv("CRUXEVAL_MAX_WORKERS", "32"))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                fn,
                client,
                i,
                cache,
                query,
                n,
                model,
                stop,
                **generation_args,
            )
            for i, query in enumerate(queries)
        ]
        results_with_id = [future.result() for future in futures]
    results_with_id.sort()
    results = [i[1] for i in results_with_id]

    if _empty_completions:
        logger.warning(
            "%d/%d completions carried no answer and are scored as misses (by "
            "finish_reason: %s). pass@k is a floor, not the model's score, until "
            "this is 0.",
            sum(_empty_completions.values()),
            len(queries) * n,
            dict(_empty_completions),
        )

    with open(cache_dir_tmp, "w") as f:
        json.dump(cache, f)
    os.rename(cache_dir, cache_dir_bak)
    os.rename(cache_dir_tmp, cache_dir)
    os.remove(cache_dir_bak)

    # parse the output
    gens = [i[1] for i in results]
    return [[(extraction_fn(i), i) for i in r] for r in gens]


# direct output prompt
def prompt_direct_output(
    client, i, cache, gpt_query, n, model, stop, **generation_args
):
    return prompt_openai_general(
        client,
        make_direct_output_prompt,
        i,
        cache,
        gpt_query,
        n,
        model,
        stop,
        **generation_args,
    )


def batch_prompt_direct_output(client, queries, n, model, stop, **generation_args):
    return batch_prompt(
        prompt_direct_output,
        extract_answer_direct_output,
        client,
        queries,
        n,
        model,
        stop,
        **generation_args,
    )


# cot output prompt
def prompt_cot_output(client, i, cache, gpt_query, n, model, stop, **generation_args):
    return prompt_openai_general(
        client,
        make_cot_output_prompt,
        i,
        cache,
        gpt_query,
        n,
        model,
        stop,
        **generation_args,
    )


def batch_prompt_cot_output(client, queries, n, model, stop, **generation_args):
    return batch_prompt(
        prompt_cot_output,
        extract_answer_cot_output,
        client,
        queries,
        n,
        model,
        stop,
        **generation_args,
    )


# direct input prompt
def prompt_direct_input(client, i, cache, gpt_query, n, model, stop, **generation_args):
    return prompt_openai_general(
        client,
        make_direct_input_prompt,
        i,
        cache,
        gpt_query,
        n,
        model,
        stop,
        **generation_args,
    )


def batch_prompt_direct_input(client, queries, n, model, stop, **generation_args):
    return batch_prompt(
        prompt_direct_input,
        extract_answer_direct_input,
        client,
        queries,
        n,
        model,
        stop,
        **generation_args,
    )


# cot input prompt
def prompt_cot_input(client, i, cache, gpt_query, n, model, stop, **generation_args):
    return prompt_openai_general(
        client,
        make_cot_input_prompt,
        i,
        cache,
        gpt_query,
        n,
        model,
        stop,
        **generation_args,
    )


def batch_prompt_cot_input(client, queries, n, model, stop, **generation_args):
    return batch_prompt(
        prompt_cot_input,
        extract_answer_cot_input,
        client,
        queries,
        n,
        model,
        stop,
        **generation_args,
    )
