"""Experiment harness: every embedding model x every LLM x every test case.

Usage::

    python -m rag_flint.experiment --isde-path data/ISDE_4.5.txt --output-dir results/raw

Each (embedder, generator) combination is written to its own JSON file, so a
crashed or out-of-memory run never loses earlier results.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import List, Optional

from .chunking import load_chunks
from .config import (DEFAULT_ISDE_PATH, DEFAULT_OUTPUT_DIR, EMBEDDING_MODELS,
                     GENERATIVE_MODELS, TEST_CASES, TOP_K)
from .generation import deduplicate_frames, extract_generation_attempts_for_test_case, load_generator
from .retrieval import build_index, embed_bert_with_pooling, load_embedding_model


def hf_login() -> None:
    """Log in to Hugging Face with a token from the environment or Colab secrets.

    Never hard-code the token: set ``HF_TOKEN`` as an environment variable or
    as a Colab secret instead.
    """
    from huggingface_hub import login

    token = os.environ.get("HF_TOKEN")
    if token is None:
        try:
            from google.colab import userdata  # type: ignore
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass
    if token:
        login(token=token)
    else:
        print("No HF_TOKEN found: gated models (e.g. Mistral) will fail to download.")


def result_filename(embed_model: str, gen_model: str) -> str:
    return f"embed_{embed_model.replace('/', '_')}__gen_{gen_model.replace('/', '_')}.json"


def _free_gpu_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_experiments(
    isde_path: Path = DEFAULT_ISDE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    embedding_models: Optional[List[str]] = None,
    generative_models: Optional[List[str]] = None,
    test_cases: Optional[List[dict]] = None,
    top_k: int = TOP_K,
) -> None:
    import torch

    embedding_models = embedding_models or EMBEDDING_MODELS
    generative_models = generative_models or GENERATIVE_MODELS
    test_cases = test_cases or TEST_CASES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(isde_path.read_text(encoding="utf-8"))
    print(f"Created {len(chunks)} clean, article-aware chunks.")
    chunk_texts = [c["full_text"] for c in chunks]

    for embed_name in embedding_models:
        print("=" * 80 + f"\nEmbedding model: {embed_name}\n" + "=" * 80)
        embed_model, embed_tokenizer = load_embedding_model(embed_name, device)
        embeddings = embed_bert_with_pooling(chunk_texts, embed_model, embed_tokenizer)
        index = build_index(embeddings)

        for gen_name in generative_models:
            print("-" * 80 + f"\nGenerative model: {gen_name}\n" + "-" * 80)
            results = []
            generator = model = tokenizer = None
            try:
                start = time.time()
                generator, model, tokenizer = load_generator(gen_name)
                print(f"  > Model loaded in {time.time() - start:.1f}s")

                for case in test_cases:
                    print(f"\n  {case['id']}: {case['description']}")
                    start = time.time()
                    retrieved, attempts = extract_generation_attempts_for_test_case(
                        case["description"], generator, tokenizer, chunks, embeddings, index,
                        embed_model, embed_tokenizer, top_k=top_k,
                    )
                    frames, scores = [], []
                    for attempt in attempts:
                        for frame in attempt["parsed_frames"]:
                            frames.append(frame)
                            scores.append(attempt["retrieval_score"])
                    unique = deduplicate_frames(frames, scores)
                    print(f"  > Unique frames: {len(unique)}")

                    results.append({
                        "embedding_model": embed_name,
                        "generative_model": gen_name,
                        "test_case_id": case["id"],
                        "test_case_description": case["description"],
                        "execution_time_seconds": time.time() - start,
                        "retrieved_chunks": retrieved,
                        "generation_attempts": attempts,
                        "final_unique_frames": unique,
                        "error": None,
                    })
            except Exception as exc:  # e.g. out of GPU memory: log and continue
                print(f"ERROR with {gen_name}: {exc}")
                results.append({"embedding_model": embed_name, "generative_model": gen_name, "error": str(exc)})
            finally:
                path = output_dir / result_filename(embed_name, gen_name)
                path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  > Saved {path}")
                del generator, model, tokenizer
                _free_gpu_memory()

        del embed_model, embed_tokenizer, embeddings, index
        _free_gpu_memory()

    print("\nAll model combinations done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG + FLINT experiments.")
    parser.add_argument("--isde-path", type=Path, default=DEFAULT_ISDE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-models", nargs="+", help="Override the embedding models")
    parser.add_argument("--generative-models", nargs="+", help="Override the generative models")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_login()
    run_experiments(
        isde_path=args.isde_path,
        output_dir=args.output_dir,
        embedding_models=args.embedding_models,
        generative_models=args.generative_models,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
