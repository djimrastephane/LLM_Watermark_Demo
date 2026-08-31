"""Next-token probability extraction and JSON export.

Educational simulation of keyed statistical text watermarking. This module extracts REAL
next-token probabilities from an open-weight model (Qwen). No watermarking is applied here --
that happens later, in the Mathematica notebook, as a clearly-labelled simulation.
"""
import json
from pathlib import Path

import torch


def get_next_token_distribution(model, tokenizer, prompt, top_k=15):
    """Runs one forward pass and returns the top-k next-token candidates for `prompt`.

    Returns a dict with "input_token_count" and "candidates" (a list of dicts with rank,
    token_id, token, probability, logit -- probabilities are NOT renormalized, since they are
    only the top-k subset of the full vocabulary distribution).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]
    probabilities = torch.softmax(next_token_logits, dim=-1)

    top_probs, top_ids = torch.topk(probabilities, top_k)
    top_logits = next_token_logits[top_ids]

    candidates = []
    for rank, (token_id, prob, logit) in enumerate(
        zip(top_ids.tolist(), top_probs.tolist(), top_logits.tolist()), start=1
    ):
        candidates.append({
            "rank": rank,
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "probability": prob,
            "logit": logit,
        })

    return {"input_token_count": input_token_count, "candidates": candidates}


def generate_sequence(model, tokenizer, prompt, num_steps=8, top_k=15):
    """Greedily extends `prompt` by `num_steps` real tokens, one forward pass per step.

    At each step the model's own top-1 candidate is appended to the running text (this is
    real, deterministic generation -- not random sampling). The full top-k distribution at
    each step is also recorded, so a downstream tool (e.g. the Mathematica notebook) can
    compare what the model actually chose against what a simulated keyed watermark would
    have preferred at that same real decision point.

    Returns a dict with "seed_prompt" and "steps" (a list of dicts with "step",
    "context_before", "chosen_token", and "candidates").
    """
    steps = []
    context = prompt

    for step in range(num_steps):
        dist = get_next_token_distribution(model, tokenizer, context, top_k=top_k)
        chosen_token = dist["candidates"][0]["token"]
        steps.append({
            "step": step + 1,
            "context_before": context,
            "chosen_token": chosen_token,
            "candidates": dist["candidates"],
        })
        context += chosen_token

    return {"seed_prompt": prompt, "steps": steps}


def extract_all_prompts(model, tokenizer, prompts, top_k=15):
    """Runs get_next_token_distribution for every (prompt_id, prompt_text) pair in `prompts`.

    `prompts` is a list of (prompt_id, prompt_text) tuples. Returns a list of per-prompt result
    dicts ready to embed in the exported JSON's "prompts" array.
    """
    results = []
    for prompt_id, prompt_text in prompts:
        dist = get_next_token_distribution(model, tokenizer, prompt_text, top_k=top_k)
        results.append({
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "input_token_count": dist["input_token_count"],
            "candidates": dist["candidates"],
        })
    return results


def export_distributions(output_path, model_name, top_k, device, prompt_results,
                          generation=None,
                          description="Real next-token probabilities extracted from an "
                                       "open-weight model for educational watermarking "
                                       "demonstrations."):
    """Writes the full result object (metadata + prompts [+ optional generation]) to
    `output_path` as UTF-8 JSON, preserving token strings (including leading whitespace)
    exactly as decoded. `generation`, if given, is the dict returned by `generate_sequence`."""
    result = {
        "metadata": {
            "model": model_name,
            "top_k": top_k,
            "device": str(device),
            "description": description,
            "watermarking": "not applied in Python extraction stage",
        },
        "prompts": prompt_results,
    }
    if generation is not None:
        result["generation"] = generation

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
