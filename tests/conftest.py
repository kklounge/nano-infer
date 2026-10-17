"""Shared pytest fixtures.

* tiny random models (Qwen2 / Qwen3 architecture, a few MB) are created on the fly, so the
  core tests run anywhere in seconds without downloading anything;
* the real Qwen2.5-0.5B-Instruct tests run only if the weights are present
  (set NANOINFER_MODEL=path, or put them in ./models/Qwen2.5-0.5B-Instruct).
"""

import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

# No GPU: run the Triton kernels in Triton's CPU interpreter, so they are still tested.
# (Must be set before triton is imported.)
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TINY_VOCAB = 512


def _candidate_model_dirs():
    env = os.environ.get("NANOINFER_MODEL")
    if env:
        yield Path(env)
    yield ROOT / "models" / "Qwen2.5-0.5B-Instruct"
    yield Path("/workspace/hf_models/Qwen2.5-0.5B-Instruct")


def find_real_model() -> Path | None:
    for p in _candidate_model_dirs():
        if (p / "config.json").exists() and any(p.glob("*.safetensors")):
            return p
    return None


def make_tiny_model(path: Path, arch: str = "qwen2", vocab_size: int = TINY_VOCAB, seed: int = 0):
    """Save a randomly initialised small HF model (float32 safetensors) to `path`."""
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
    common = dict(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                  rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=(arch == "qwen2"),
                  # larger init than usual -> peaky logits, so float rounding can't flip the argmax
                  initializer_range=0.2, eos_token_id=vocab_size - 1, bos_token_id=vocab_size - 2)
    torch.manual_seed(seed)
    if arch == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(**common))
    else:
        model = Qwen3ForCausalLM(Qwen3Config(head_dim=16, attention_bias=False, **common))
    model.save_pretrained(path, safe_serialization=True)
    return path


@pytest.fixture(scope="session")
def tiny_qwen2(tmp_path_factory) -> Path:
    return make_tiny_model(tmp_path_factory.mktemp("tiny_qwen2"), "qwen2")


@pytest.fixture(scope="session")
def tiny_qwen3(tmp_path_factory) -> Path:
    return make_tiny_model(tmp_path_factory.mktemp("tiny_qwen3"), "qwen3")


@pytest.fixture(scope="session")
def real_model() -> Path:
    p = find_real_model()
    if p is None:
        pytest.skip("real model weights not found (run scripts/download_model.py)")
    return p


@pytest.fixture(scope="session")
def tiny_with_tokenizer(tmp_path_factory, real_model) -> Path:
    """Tiny random model that shares the real Qwen tokenizer (for fast server tests)."""
    path = tmp_path_factory.mktemp("tiny_tok")
    make_tiny_model(path, "qwen2", vocab_size=151936)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if (real_model / name).exists():
            shutil.copy(real_model / name, path / name)
    return path


def hf_greedy(model, prompt_ids: list[int], max_new_tokens: int, eos_ids=None) -> list[int]:
    """Reference greedy decoding with Hugging Face (no repetition penalty, no sampling)."""
    out = model.generate(
        torch.tensor([prompt_ids]), max_new_tokens=max_new_tokens, do_sample=False,
        repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
        eos_token_id=eos_ids, pad_token_id=(eos_ids[0] if eos_ids else None),
    )[0][len(prompt_ids):].tolist()
    if eos_ids:  # cut right after the first eos, like our engine does
        for i, t in enumerate(out):
            if t in eos_ids:
                return out[:i + 1]
    return out
