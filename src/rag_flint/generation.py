"""Prompting open-source LLMs to turn law chunks into FLINT frames."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

from .config import GENERATION_KWARGS, TOP_K
from .retrieval import retrieve_chunks_with_reranking

# Fallback chat template for models that do not ship one.
PLAIN_CHAT_TEMPLATE = (
    "{% for message in messages %}{% if message['role'] == 'user' %}"
    "{{ message['content'] }}{% endif %}{% endfor %}"
)


def prepare_prompt_content(query: str, chunk: Dict[str, str]) -> str:
    """Build the zero-shot, schema-constrained prompt (in Dutch, like the law text)."""
    return f"""Jouw taak is om de WETTEKST te analyseren en de juridische regels die van toepassing zijn op het SCENARIO te extraheren. Je MOET de output formatteren als een JSON-object volgens het onderstaande SCHEMA.

    --- SCHEMA DEFINITIE ---
    Je moet een lijst van "frames" teruggeven. Elk frame in de lijst moet de volgende exacte structuur hebben:
    {{
      "actor": "STRING // Wie of wat voert de actie uit?",
      "action": "STRING // Welke handeling wordt er uitgevoerd?",
      "object": "STRING // Waarop heeft de handeling betrekking?",
      "conditions": "STRING // ALLE voorwaarden waaraan voldaan moet worden, letterlijk uit de tekst.",
      "results": "STRING // ALLE juridische gevolgen als aan de voorwaarden is voldaan, letterlijk uit de tekst."
    }}
    --- EINDE SCHEMA ---

    Hier is de taak:

    SCENARIO: "{query}"

    WETTEKST:
    {chunk['text']}

    --- FINALE INSTRUCTIE ---
    Genereer een JSON-object met een "frames"-lijst. Elk frame MOET de structuur van het hierboven gedefinieerde SCHEMA volgen.
    Extraheer de informatie UITSLUITEND uit de WETTEKST. Uw antwoord moet een neutrale, derdepersoonsanalyse van de wet zijn.
    Als de WETTEKST geen relevante informatie bevat voor het scenario, geef dan een lege lijst terug: `{{"frames": []}}`.
    """


def extract_json(text: str) -> dict:
    """Parse model output into ``{"frames": [...]}``, tolerating text around the JSON."""
    # Strategy 1: a complete {"frames": [...]} object.
    match = re.search(r'\{\s*"frames"\s*:\s*\[.*\]\s*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 2: a single frame object with all five keys.
    match = re.search(
        r'\{[^{}]*"actor"[^{}]*"action"[^{}]*"object"[^{}]*"conditions"[^{}]*"results"[^{}]*\}',
        text, re.DOTALL,
    )
    if match:
        try:
            return {"frames": [json.loads(match.group(0))]}
        except json.JSONDecodeError:
            pass

    return {"frames": []}


def deduplicate_frames(frames: List[dict], scores: List[float]) -> List[dict]:
    """Keep one frame per (action, object): the one from the best-scoring chunk."""
    groups: Dict[tuple, list] = {}
    for frame, score in zip(frames, scores):
        key = (str(frame.get("action", "")).lower().strip(), str(frame.get("object", "")).lower().strip())
        groups.setdefault(key, []).append((frame, score))

    unique = []
    for group in groups.values():
        best_frame, _ = max(
            group,
            key=lambda item: (item[1], len(str(item[0].get("conditions", ""))), len(str(item[0].get("results", "")))),
        )
        unique.append(best_frame)
    return unique


def load_generator(model_name: str):
    """Load a causal LM in 4-bit (NF4) and wrap it in a text-generation pipeline."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = PLAIN_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator, model, tokenizer


def extract_generation_attempts_for_test_case(
    query: str,
    generator,
    tokenizer,
    chunks: List[Dict[str, str]],
    embeddings,
    index,
    embed_model,
    embed_tokenizer,
    top_k: int = TOP_K,
) -> Tuple[List[Dict], List[Dict]]:
    """Retrieve chunks for a question and generate one set of frames per chunk (batched)."""
    _, scores, retrieved = retrieve_chunks_with_reranking(
        query, chunks, embeddings, index, embed_model, embed_tokenizer, k=top_k,
    )
    print(f"  > Retrieved {len(retrieved)} chunks for query: '{query[:50]}...'")
    if not retrieved:
        return [], []

    stored_chunks = [
        {"text": c["text"], "article": c["article"], "retrieval_score": float(s)}
        for c, s in zip(retrieved, scores)
    ]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prepare_prompt_content(query, chunk)}],
            tokenize=False, add_generation_prompt=True,
        )
        for chunk in retrieved
    ]
    outputs = generator(prompts, pad_token_id=tokenizer.eos_token_id, **GENERATION_KWARGS)

    attempts = []
    for prompt, output, chunk, score in zip(prompts, outputs, retrieved, scores):
        raw_text = ""
        if output and isinstance(output, list) and "generated_text" in output[0]:
            raw_text = output[0]["generated_text"].replace(prompt, "").strip()
        attempts.append({
            "source_article": chunk["article"],
            "retrieval_score": float(score),
            "raw_model_output": raw_text,
            "parsed_frames": extract_json(raw_text).get("frames", []),
        })
    return stored_chunks, attempts
