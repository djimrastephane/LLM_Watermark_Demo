"""Schema and sanity checks for the exported next-token distribution JSON.

Educational simulation of keyed statistical text watermarking. These checks only validate the
structure/values of REAL next-token probabilities exported from the Python extraction stage --
no watermarking logic lives here.
"""

REQUIRED_TOP_LEVEL_KEYS = {"metadata", "prompts"}
REQUIRED_CANDIDATE_FIELDS = {"rank", "token_id", "token", "probability", "logit"}


def check_top_level_keys(result):
    missing = REQUIRED_TOP_LEVEL_KEYS - set(result.keys())
    if missing:
        raise ValueError(f"Missing required top-level keys: {missing}")
    return True


def check_prompts_is_list(result):
    if not isinstance(result["prompts"], list):
        raise ValueError("'prompts' must be a list")
    if len(result["prompts"]) == 0:
        raise ValueError("'prompts' must not be empty")
    return True


def check_candidate_fields(candidate):
    missing = REQUIRED_CANDIDATE_FIELDS - set(candidate.keys())
    if missing:
        raise ValueError(f"Candidate missing required fields: {missing}")
    return True


def check_probability_range(candidate):
    p = candidate["probability"]
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Probability out of range [0, 1]: {p}")
    return True


def check_ranks_sequential(candidates):
    ranks = [c["rank"] for c in candidates]
    expected = list(range(1, len(candidates) + 1))
    if ranks != expected:
        raise ValueError(f"Ranks are not sequential starting at 1: {ranks}")
    return True


def check_probabilities_descending(candidates):
    probs = [c["probability"] for c in candidates]
    if probs != sorted(probs, reverse=True):
        raise ValueError(f"Probabilities are not in descending order: {probs}")
    return True


def check_prompt_entry(prompt_entry, expected_top_k=None):
    """Runs every candidate-level check for one prompt entry, plus an optional top-k count
    check."""
    if "candidates" not in prompt_entry:
        raise ValueError("Prompt entry missing 'candidates'")
    candidates = prompt_entry["candidates"]

    if expected_top_k is not None and len(candidates) != expected_top_k:
        raise ValueError(
            f"Expected {expected_top_k} candidates, found {len(candidates)} "
            f"for prompt_id={prompt_entry.get('prompt_id')!r}"
        )

    for candidate in candidates:
        check_candidate_fields(candidate)
        check_probability_range(candidate)

    check_ranks_sequential(candidates)
    check_probabilities_descending(candidates)
    return True


def validate_export(result):
    """Runs the full validation suite against one exported result object. Raises ValueError
    on the first failure; returns True if everything passes."""
    check_top_level_keys(result)
    check_prompts_is_list(result)

    expected_top_k = result.get("metadata", {}).get("top_k")
    for prompt_entry in result["prompts"]:
        check_prompt_entry(prompt_entry, expected_top_k=expected_top_k)

    return True
