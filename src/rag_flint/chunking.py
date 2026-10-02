"""Article-aware chunking and cleaning of the ISDE regulation text."""

import re
from typing import Dict, List

# Chunks that mention none of these terms carry no subsidy-relevant rule.
RELEVANT_TERMS = [
    "subsidie", "investering", "warmtepomp", "isolatie",
    "zonneboiler", "windturbine", "warmtenet", "eigenaar-bewoner",
    "voorwaarden", "bedraagt", "€", "vierkante meter", "vermogen",
    "toegewezen",
]

# Chunks containing these are metadata or cross-references, not rules.
METADATA_MARKERS = ["geldend van", "datum", "t/m", "zie:", "zie artikel"]


def _make_chunk(text: str, article: str) -> Dict[str, str]:
    return {"text": text, "article": article, "full_text": f"Artikel {article}: {text}"}


def chunk_text_articles(text: str) -> List[Dict[str, str]]:
    """Split the law text into chunks that keep a reference to their article.

    A new chunk starts at every ``Artikel x.y.z`` header, and long sections are
    split further at sub-points (``1.``, ``1°.``, ``a.``).
    """
    lines = text.splitlines()
    chunks: List[Dict[str, str]] = []
    current_article = None
    current_section: List[str] = []

    for line in lines:
        article_match = re.match(r"^Artikel\s+(\d+\.\d+\.\d+)", line)
        if article_match:
            # Close the previous article.
            if current_section and current_article:
                chunk_text = " ".join(current_section).strip()
                if len(chunk_text) > 40:
                    chunks.append(_make_chunk(chunk_text, current_article))
            current_article = article_match.group(1)
            current_section = [line]
        elif re.match(r"^\s*(\d+°?\.|\w\.)\s+", line):
            # Sub-point: start a new chunk if the current one is long enough.
            if current_section and len(" ".join(current_section)) > 100:
                chunk_text = " ".join(current_section).strip()
                if current_article:
                    chunks.append(_make_chunk(chunk_text, current_article))
                current_section = [line]
            else:
                current_section.append(line.strip())
        elif line.strip():
            current_section.append(line.strip())

    # Close the last section.
    if current_section and current_article:
        chunk_text = " ".join(current_section).strip()
        if len(chunk_text) > 40:
            chunks.append(_make_chunk(chunk_text, current_article))

    return chunks


def clean_isde_chunks(chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep only chunks with substantive legal content about the subsidy."""
    filtered = []
    for chunk in chunks:
        text = chunk["text"].strip().lower()
        if any(marker in text for marker in METADATA_MARKERS):
            continue
        # Skip bare definitions unless they are about the subsidy itself.
        if re.match(r"^(een.*natuurlijke persoon.*|.*wordt verstaan onder.*)$", text):
            if "subsidie" not in text and "investering" not in text:
                continue
        if any(term in text for term in RELEVANT_TERMS):
            filtered.append(chunk)
    return filtered


def load_chunks(law_text: str) -> List[Dict[str, str]]:
    """Chunk and clean the law text in one step."""
    return clean_isde_chunks(chunk_text_articles(law_text))
