"""Tests for the exported next-token distribution JSON schema.

These tests use a small mocked result structure and never load a model.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation import (
    validate_export,
    check_top_level_keys,
    check_prompts_is_list,
    check_candidate_fields,
    check_probability_range,
    check_ranks_sequential,
    check_probabilities_descending,
)


def make_candidate(rank, token_id, token, probability, logit):
    return {
        "rank": rank,
        "token_id": token_id,
        "token": token,
        "probability": probability,
        "logit": logit,
    }


def make_mock_result(top_k=3):
    candidates = [
        make_candidate(1, 101, " circulated", 0.42, 8.1),
        make_candidate(2, 102, " pulled", 0.31, 7.6),
        make_candidate(3, 103, " prepared", 0.10, 6.2),
    ][:top_k]
    return {
        "metadata": {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "top_k": top_k,
            "device": "cpu",
            "description": "mock",
            "watermarking": "not applied in Python extraction stage",
        },
        "prompts": [
            {
                "prompt_id": "P01",
                "prompt": "After reaching TD, the drilling team",
                "input_token_count": 8,
                "candidates": candidates,
            }
        ],
    }


def test_valid_mock_result_passes():
    result = make_mock_result()
    assert validate_export(result) is True


def test_required_top_level_keys_present():
    result = make_mock_result()
    assert check_top_level_keys(result) is True


def test_missing_top_level_key_raises():
    result = make_mock_result()
    del result["metadata"]
    with pytest.raises(ValueError):
        check_top_level_keys(result)


def test_prompts_is_list():
    result = make_mock_result()
    assert check_prompts_is_list(result) is True


def test_prompts_not_a_list_raises():
    result = make_mock_result()
    result["prompts"] = "not a list"
    with pytest.raises(ValueError):
        check_prompts_is_list(result)


def test_empty_prompts_list_raises():
    result = make_mock_result()
    result["prompts"] = []
    with pytest.raises(ValueError):
        check_prompts_is_list(result)


def test_each_prompt_has_candidates():
    result = make_mock_result()
    for prompt_entry in result["prompts"]:
        assert "candidates" in prompt_entry
        assert len(prompt_entry["candidates"]) > 0


def test_candidate_required_fields_present():
    candidate = make_candidate(1, 101, " circulated", 0.42, 8.1)
    assert check_candidate_fields(candidate) is True


def test_candidate_missing_field_raises():
    candidate = make_candidate(1, 101, " circulated", 0.42, 8.1)
    del candidate["logit"]
    with pytest.raises(ValueError):
        check_candidate_fields(candidate)


def test_probabilities_are_valid_range():
    candidate = make_candidate(1, 101, " circulated", 0.42, 8.1)
    assert check_probability_range(candidate) is True


@pytest.mark.parametrize("bad_prob", [-0.1, 1.1, 2.0])
def test_probability_out_of_range_raises(bad_prob):
    candidate = make_candidate(1, 101, " circulated", bad_prob, 8.1)
    with pytest.raises(ValueError):
        check_probability_range(candidate)


def test_ranks_are_sequential():
    candidates = [
        make_candidate(1, 101, " circulated", 0.5, 8.0),
        make_candidate(2, 102, " pulled", 0.3, 7.0),
        make_candidate(3, 103, " prepared", 0.2, 6.0),
    ]
    assert check_ranks_sequential(candidates) is True


def test_ranks_not_sequential_raises():
    candidates = [
        make_candidate(1, 101, " circulated", 0.5, 8.0),
        make_candidate(3, 103, " prepared", 0.3, 7.0),
    ]
    with pytest.raises(ValueError):
        check_ranks_sequential(candidates)


def test_probabilities_are_descending():
    candidates = [
        make_candidate(1, 101, " circulated", 0.5, 8.0),
        make_candidate(2, 102, " pulled", 0.3, 7.0),
        make_candidate(3, 103, " prepared", 0.2, 6.0),
    ]
    assert check_probabilities_descending(candidates) is True


def test_probabilities_not_descending_raises():
    candidates = [
        make_candidate(1, 101, " circulated", 0.2, 8.0),
        make_candidate(2, 102, " pulled", 0.5, 7.0),
    ]
    with pytest.raises(ValueError):
        check_probabilities_descending(candidates)


def test_expected_top_k_mismatch_raises():
    result = make_mock_result(top_k=3)
    with pytest.raises(ValueError):
        from src.validation import check_prompt_entry
        check_prompt_entry(result["prompts"][0], expected_top_k=15)
