"""Device selection and model/tokenizer loading utilities.

Educational simulation of keyed statistical text watermarking. This module only loads an
open-weight model (Qwen) to obtain real next-token probabilities -- it has nothing to do with
Anthropic's proprietary watermarking implementation.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    """Picks the best available device: MPS (Apple Silicon) > CUDA > CPU. Never hard-codes CUDA."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(model_name, device=None):
    """Loads a causal LM and its tokenizer from Hugging Face (local cache after first
    download -- no remote inference API is used at generation time)."""
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # float16/bfloat16 on GPU-like devices for speed/memory; float32 on CPU for stability.
    dtype = torch.float32
    if device.type in ("mps", "cuda"):
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()

    return model, tokenizer, device
