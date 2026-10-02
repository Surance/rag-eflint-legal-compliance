"""Evaluation metrics for retrieval and generated FLINT frames.

All functions take the raw result records written by ``rag_flint.experiment``
(one dict per embedder x generator x test case).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from ..config import FLINT_KEYS, TEST_CASE_KEYWORDS


# --------------------------------------------------------------------------- loading
def load_results(paths: Iterable[Path]) -> List[dict]:
    """Load and concatenate result files, skipping records that only hold an error."""
    records = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        records.extend(r for r in (data if isinstance(data, list) else [data]) if "test_case_id" in r)
    return records


def short_name(model: str) -> str:
    return model.split("/")[-1]


# --------------------------------------------------------------------------- retrieval
def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of all relevant chunks that appear in the top ``k`` retrieved chunks."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def recall_table(retrieved: Dict[str, Dict[str, List[str]]], ground_truth: Dict[str, List[str]],
                 ks=(3, 5, 10)) -> pd.DataFrame:
    """Recall@k per embedder and test case. ``retrieved[embedder][test_case] = [chunk ids]``."""
    rows = []
    for embedder, cases in retrieved.items():
        for tc, ids in cases.items():
            if tc in ground_truth:
                rows.append({"Embedder": embedder, "Test Case": tc,
                             **{f"Recall@{k}": recall_at_k(ids, ground_truth[tc], k) for k in ks}})
    return pd.DataFrame(rows)


def top1_retrieval_scores(records: List[dict]) -> pd.DataFrame:
    rows = [{"embedder": r["embedding_model"],
             "top_1_score": r["retrieved_chunks"][0]["retrieval_score"] if r.get("retrieved_chunks") else 0.0}
            for r in records]
    return pd.DataFrame(rows)


def embedder_overlap(records: List[dict]) -> pd.Series:
    """Jaccard overlap of the retrieved chunk texts of the two embedders, per test case."""
    sets = defaultdict(lambda: defaultdict(set))
    for r in records:
        for chunk in r.get("retrieved_chunks", []):
            if chunk.get("text"):
                sets[r["test_case_id"]][r["embedding_model"]].add(chunk["text"])
    scores = {}
    for tc, by_embedder in sets.items():
        if len(by_embedder) == 2:
            a, b = by_embedder.values()
            scores[tc] = len(a & b) / len(a | b) if a | b else 0.0
    return pd.Series(scores).sort_index()


# --------------------------------------------------------------------------- frames
def frame_counts(records: List[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"embedder": r["embedding_model"], "generator": r["generative_model"],
                          "test_case": r["test_case_id"], "num_frames": len(r.get("final_unique_frames", []))}
                         for r in records])


def schema_adherence(records: List[dict]) -> pd.DataFrame:
    """Count frames that contain all five FLINT keys, per generator."""
    counts = defaultdict(lambda: {"adhering": 0, "non_adhering": 0})
    for r in records:
        for frame in r.get("final_unique_frames", []):
            ok = isinstance(frame, dict) and set(FLINT_KEYS).issubset(frame)
            counts[r["generative_model"]]["adhering" if ok else "non_adhering"] += 1
    return pd.DataFrame.from_dict(counts, orient="index")


def language_distribution(records: List[dict]) -> pd.DataFrame:
    """Detected language (Dutch / English / other) of every schema-adhering frame, per generator."""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0  # reproducible detection
    lang_map = {"nl": "dutch", "en": "english"}
    counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        for frame in r.get("final_unique_frames", []):
            if not (isinstance(frame, dict) and set(FLINT_KEYS).issubset(frame)):
                continue
            text = " ".join(str(v) for v in frame.values() if isinstance(v, str))
            try:
                lang = lang_map.get(detect(text), "mixed/other") if text.strip() else "unknown"
            except LangDetectException:
                lang = "unknown"
            counts[r["generative_model"]][lang] += 1
    return pd.DataFrame.from_dict(counts, orient="index").fillna(0).astype(int)


def grounding_score(frame: dict, retrieved_chunks: List[dict]) -> float:
    """Jaccard similarity between the words of a frame and of its retrieved chunks.

    A rough proxy for factual grounding: frames that reuse the wording of the
    law score higher than frames with invented content.
    """
    frame_terms = set(re.findall(r"\b\w+\b", " ".join(str(frame.get(k, "")) for k in FLINT_KEYS).lower()))
    chunk_terms = set()
    for chunk in retrieved_chunks:
        chunk_terms.update(re.findall(r"\b\w+\b", chunk.get("text", "").lower()))
    if not frame_terms or not chunk_terms:
        return 0.0
    return len(frame_terms & chunk_terms) / len(frame_terms | chunk_terms)


def grounding_scores(records: List[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"generator": r["generative_model"],
                          "grounding": grounding_score(f, r.get("retrieved_chunks", []))}
                         for r in records for f in r.get("final_unique_frames", [])])


def frame_quality(records: List[dict]) -> pd.DataFrame:
    """Verbosity, concreteness, keyword match, article references, completeness and run time."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["embedding_model"], r["generative_model"])].append(r)

    rows = []
    for (embedder, generator), recs in groups.items():
        frames = [(f, r["test_case_id"]) for r in recs for f in r.get("final_unique_frames", [])]
        n = len(frames)
        words = numbers = currency = articles = complete = 0
        keyword_scores = []
        for frame, tc in frames:
            text = " ".join(str(v) for v in frame.values() if v is not None).lower()
            words += len(text.split())
            keywords = TEST_CASE_KEYWORDS.get(tc, [])
            keyword_scores.append(sum(k in text for k in keywords) / len(keywords) if keywords else 0)
            numbers += bool(re.search(r"\d", text))
            currency += "€" in text
            articles += "artikel" in text or "lid" in text
            complete += all(frame.get(k) for k in FLINT_KEYS)
        times = [r.get("execution_time_seconds", 0) for r in recs]
        pct = (lambda x: 100 * x / n) if n else (lambda x: 0.0)
        rows.append({
            "Embedding Model": short_name(embedder),
            "Generative Model": short_name(generator),
            "Frames": n,
            "Avg. Words": words / n if n else 0.0,
            "Keyword Match (%)": 100 * sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0,
            "Frames with Numbers (%)": pct(numbers),
            "Frames with Currency (%)": pct(currency),
            "Article References (%)": pct(articles),
            "Field Fill Rate (%)": pct(complete),
            "Avg. Time per Case (s)": sum(times) / len(times) if times else 0.0,
        })
    return pd.DataFrame(rows).round(2)
