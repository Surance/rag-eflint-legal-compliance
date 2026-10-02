"""Tests for the parts of the pipeline that run without a GPU or model downloads."""

from rag_flint.chunking import chunk_text_articles, clean_isde_chunks
from rag_flint.evaluation.metrics import grounding_score, recall_at_k
from rag_flint.generation import deduplicate_frames, extract_json
from rag_flint.retrieval import extract_keywords, rerank_score

LAW = """Artikel 4.5.1. Begripsbepalingen
In deze titel wordt verstaan onder warmtepomp: een toestel dat warmte onttrekt aan de omgeving.
Artikel 4.5.2. Subsidiabele activiteiten
1. De minister kan op aanvraag subsidie verstrekken aan een eigenaar-bewoner voor een investering in een warmtepomp, een zonneboiler of isolatiemaatregelen in de eigen woning.
a. de subsidie bedraagt € 500 per warmtepomp met een vermogen tot 10 kW, en meer bij een groter vermogen dan dat.
Artikel 4.5.3. Geldend van 1 januari 2024 t/m heden
Zie artikel 4.5.2 voor de voorwaarden.
"""


def test_chunks_keep_article_reference():
    chunks = chunk_text_articles(LAW)
    assert {c["article"] for c in chunks} == {"4.5.1", "4.5.2", "4.5.3"}
    assert all(c["full_text"].startswith(f"Artikel {c['article']}: ") for c in chunks)


def test_long_sections_are_split_at_sub_points():
    chunks = [c for c in chunk_text_articles(LAW) if c["article"] == "4.5.2"]
    assert len(chunks) == 2
    assert chunks[1]["text"].startswith("a. de subsidie bedraagt")


def test_cleaning_drops_metadata_and_bare_definitions():
    cleaned = clean_isde_chunks(chunk_text_articles(LAW))
    assert {c["article"] for c in cleaned} == {"4.5.2"}


def test_extract_json_full_object_with_surrounding_text():
    out = 'Hier is het antwoord: {"frames": [{"actor": "eigenaar-bewoner", "action": "aanvragen", "object": "subsidie", "conditions": "x", "results": "y"}]} Klaar.'
    assert extract_json(out)["frames"][0]["actor"] == "eigenaar-bewoner"


def test_extract_json_single_frame_fallback():
    out = '```{"actor": "a", "action": "b", "object": "c", "conditions": "d", "results": "e"}```'
    assert extract_json(out) == {"frames": [{"actor": "a", "action": "b", "object": "c", "conditions": "d", "results": "e"}]}


def test_extract_json_returns_empty_list_on_garbage():
    assert extract_json("geen json hier") == {"frames": []}


def test_deduplicate_keeps_frame_from_best_chunk():
    frames = [
        {"action": "Installeren", "object": "Warmtepomp", "conditions": "kort"},
        {"action": "installeren ", "object": "warmtepomp", "conditions": "langere voorwaarden"},
        {"action": "aanvragen", "object": "subsidie"},
    ]
    unique = deduplicate_frames(frames, [0.5, 0.9, 0.7])
    assert len(unique) == 2
    assert frames[1] in unique


def test_keywords_include_numbers_with_units():
    keywords = extract_keywords("Ik heb 30 m² vloerisolatie en een warmtepomp van 12 kW")
    assert {"vloerisolatie", "warmtepomp", "30 m²", "12 kw"} <= set(keywords)
    assert "een" not in keywords


def test_rerank_rewards_keyword_and_legal_term_overlap():
    keywords = ["warmtepomp"]
    relevant = rerank_score(0.5, "De subsidie voor een warmtepomp bedraagt", keywords)
    irrelevant = rerank_score(0.5, "Iets heel anders", keywords)
    assert relevant > irrelevant


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c", "d"], ["b", "d", "x", "y"], k=2) == 0.25
    assert recall_at_k(["a"], [], k=1) == 0.0


def test_grounding_score_is_word_jaccard():
    frame = {"actor": "eigenaar", "action": "aanvragen", "object": "", "conditions": "", "results": ""}
    assert grounding_score(frame, [{"text": "eigenaar aanvragen subsidie"}]) == 2 / 3
