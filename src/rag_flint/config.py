"""Experiment configuration: models, test cases and default settings."""

from pathlib import Path

# Default location of the ISDE regulation text (see data/README.md).
DEFAULT_ISDE_PATH = Path("data/ISDE_4.5.txt")
DEFAULT_OUTPUT_DIR = Path("results/raw")

# Number of chunks passed to the generator per question.
TOP_K = 10
# Number of semantic candidates that are re-ranked before taking the top K.
RERANK_POOL = 30

EMBEDDING_MODELS = [
    "Gerwin/legal-bert-dutch-english",
    "pdelobelle/robbert-v2-dutch-base",
]

GENERATIVE_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "BramVanroy/GEITje-7B-ultra",
    "ReBatch/Llama-3-8B-dutch",
    "BramVanroy/fietje-2b-chat",
    "ArliAI/Llama-3.1-8B-ArliAI-Formax-v1.0",
]

# Generation settings used for every model.
GENERATION_KWARGS = {
    "max_new_tokens": 384,
    "temperature": 0.2,
    "do_sample": True,
    "top_p": 0.95,
    "batch_size": 4,
}

# Realistic citizen questions about the ISDE subsidy (in Dutch, like the law).
TEST_CASES = [
    {
        "id": "TC01",
        "description": (
            "Ik heb in mijn koopwoning de vloerisolatie met biobased isolatiemateriaal over een "
            "oppervlakte van 30 m² laten installeren. De werkzaamheden zijn vorige maand afgerond. "
            "Wat is het te verwachten subsidiebedrag?"
        ),
    },
    {
        "id": "TC02",
        "description": (
            "Ik wil een lucht-waterwarmtepomp met een thermisch vermogen van 12 kW en een A+++ "
            "energielabel installeren in mijn woning. De installatie vindt plaats in augustus 2024. "
            "Hoe wordt de subsidie berekend?"
        ),
    },
    {
        "id": "TC03",
        "description": (
            "Kom ik in aanmerking voor subsidie als ik een zonneboiler laat installeren met een "
            "apertuuroppervlakte van 12 vierkante meter? Het is een zonneboiler met "
            "energie-efficiëntieklasse A."
        ),
    },
    {
        "id": "TC04",
        "description": (
            "Mijn huis is recent aangesloten op het warmtenet en mijn gasmeter is vorige week "
            "verwijderd. Ik wil nu ook subsidie aanvragen voor de eenmalige aanschaf van een nieuwe "
            "elektrische kookvoorziening. Is dit mogelijk en wat zijn de voorwaarden?"
        ),
    },
]

# Keywords per test case, used by the keyword-match quality metric.
TEST_CASE_KEYWORDS = {
    "TC01": ["vloerisolatie", "biobased", "30", "subsidiebedrag"],
    "TC02": ["lucht-waterwarmtepomp", "12 kw", "a+++", "berekend"],
    "TC03": ["zonneboiler", "apertuuroppervlakte", "12", "vierkante meter"],
    "TC04": ["warmtenet", "gasmeter", "elektrische kookvoorziening", "voorwaarden"],
}

FLINT_KEYS = ("actor", "action", "object", "conditions", "results")
