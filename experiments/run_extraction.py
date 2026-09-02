"""Runs the entropy/watermark investigation's larger prompt set through the SAME extraction
code the demo notebook uses (src/model_utils.py, src/extraction.py -- unmodified, unchanged).
Writes to data/processed/entropy_experiment.json, a separate file from the demo's
next_token_distributions.json -- the existing demo notebook and its data are untouched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_utils import load_model_and_tokenizer
from src.extraction import extract_all_prompts, export_distributions
from src.validation import validate_export

from prompts_entropy_investigation import ALL_PROMPTS

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K = 15
OUTPUT_PATH = str(Path(__file__).resolve().parent.parent / "data" / "processed" / "entropy_experiment.json")


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME)
    print(f"Device: {device}")
    print(f"Prompts: {len(ALL_PROMPTS)}")

    prompt_results = extract_all_prompts(model, tokenizer, ALL_PROMPTS, top_k=TOP_K)

    result = export_distributions(
        OUTPUT_PATH, MODEL_NAME, TOP_K, device, prompt_results,
        description="Real next-token probabilities for the entropy/watermark investigation's "
                     "larger (80-prompt, 4-category) set -- separate from the demo's 12 "
                     "curated prompts.",
    )
    validate_export(result)
    print(f"Wrote {OUTPUT_PATH} ({len(result['prompts'])} prompts, all validation checks passed)")


if __name__ == "__main__":
    main()
