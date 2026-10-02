"""Embedding, FAISS indexing and hybrid re-ranking of law chunks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from .config import RERANK_POOL, TOP_K

if TYPE_CHECKING:  # heavy dependencies are only imported when used
    import faiss

STOPWORDS = {
    "de", "het", "een", "en", "van", "voor", "in", "op", "aan", "binnen",
    "is", "zijn", "wordt", "werden", "was", "waren", "heeft", "hebben",
}

LEGAL_TERMS = [
    "subsidie", "voorwaarden", "investering", "eigenaar-bewoner",
    "bedraagt", "indien", "betreft", "aanvraag",
]

# Weights of the re-ranking score.
W_SEMANTIC, W_KEYWORD, W_LEGAL = 0.6, 0.3, 0.1


def load_embedding_model(model_name: str, device: str):
    """Load a Hugging Face encoder and its tokenizer."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, tokenizer


def embed_bert_with_pooling(texts: List[str], model, tokenizer, batch_size: int = 16) -> np.ndarray:
    """Embed texts with mean pooling over all tokens and L2-normalise them."""
    import torch
    from tqdm.auto import tqdm

    device = next(model.parameters()).device
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = tokenizer(
            texts[i:i + batch_size],
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**batch)
            mask = batch["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = torch.sum(outputs.last_hidden_state * mask, 1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            embs.append((summed / counts).cpu().numpy())

    matrix = np.vstack(embs)
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)


def build_index(embeddings: np.ndarray) -> "faiss.Index":
    """Build an HNSW index over the (normalised) chunk embeddings."""
    import faiss

    index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
    index.add(embeddings)
    return index


def extract_keywords(query: str, min_len: int = 3) -> List[str]:
    """Extract content words and numbers-with-units (e.g. ``30 m²``) from a question."""
    tokens = re.findall(r"\b[\w°²µ%€]+\b", query.lower())
    keywords = [t for t in tokens if len(t) >= min_len and t not in STOPWORDS and not t.isdigit()]
    keywords.extend(re.findall(r"\d+\s*(?:m²|kw|maand|mnd|jaar)", query.lower()))
    return list(set(keywords))


def rerank_score(semantic: float, text: str, keywords: List[str]) -> float:
    """Combine semantic similarity, keyword overlap and legal-term density."""
    text_lower = text.lower()
    keyword_score = sum(1 for kw in keywords if kw in text_lower) / max(len(keywords), 1)
    legal_density = sum(1 for term in LEGAL_TERMS if term in text_lower) / len(LEGAL_TERMS)
    return W_SEMANTIC * semantic + W_KEYWORD * keyword_score + W_LEGAL * legal_density


def retrieve_chunks_with_reranking(
    query: str,
    chunks: List[Dict[str, str]],
    embeddings: np.ndarray,
    index: "faiss.Index",
    embed_model,
    embed_tokenizer,
    k: int = TOP_K,
    rerank_pool: int = RERANK_POOL,
) -> Tuple[List[int], List[float], List[Dict[str, str]]]:
    """Retrieve ``rerank_pool`` semantic candidates and return the best ``k`` after re-ranking."""
    q_emb = embed_bert_with_pooling([query], embed_model, embed_tokenizer)[0]
    _, ids = index.search(q_emb.reshape(1, -1), rerank_pool)
    keywords = extract_keywords(query)

    candidates = []
    for idx in ids[0]:
        if idx == -1:  # FAISS pads with -1 when it finds fewer results
            continue
        semantic = float(embeddings[idx] @ q_emb)
        score = rerank_score(semantic, chunks[idx]["text"], keywords)
        candidates.append((int(idx), score, chunks[idx]))

    candidates.sort(key=lambda c: c[1], reverse=True)
    top = candidates[:k]
    return [c[0] for c in top], [c[1] for c in top], [c[2] for c in top]
