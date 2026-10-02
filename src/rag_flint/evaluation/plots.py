"""Figures used in the thesis. Every function saves a PNG and returns the figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

SHORT_NAMES = {
    "Llama-3.1-8B-ArliAI-Formax-v1.0": "ArliAI-Formax-v1.0",
    "GEITje-7B-ultra": "GEITje-7B-ultra",
    "fietje-2b-chat": "fietje-2b-chat",
    "Llama-3-8B-dutch": "Llama3-8B-dutch",
    "Mistral-7B-Instruct-v0.1": "Mistral-7B-Instruct",
}


def _short(model: str) -> str:
    name = model.split("/")[-1]
    return SHORT_NAMES.get(name, name)


def _style():
    plt.style.use("seaborn-v0_8-paper")
    sns.set_context("talk")


def _finish(fig, ax, out: Path):
    sns.despine(ax=ax)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_schema_adherence(df: pd.DataFrame, out: Path):
    _style()
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(df))
    ax.bar(x, df["adhering"], 0.6, label="Adhering", color="#4c72b0")
    ax.bar(x, df["non_adhering"], 0.6, bottom=df["adhering"], label="Non-adhering", color="#bababa")
    ax.set_xticks(x, [_short(m) for m in df.index], rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Total number of frames")
    ax.set_title("Frame schema adherence per model", pad=20)
    ax.legend()
    return _finish(fig, ax, out)


def plot_language_distribution(df: pd.DataFrame, out: Path):
    _style()
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for col, color in zip(["dutch", "english", "mixed/other"], sns.color_palette("viridis", 3)):
        values = df[col].values if col in df else np.zeros(len(df))
        ax.bar(x, values, 0.6, bottom=bottom, label=col.capitalize(), color=color)
        bottom += values
    ax.set_xticks(x, [_short(m) for m in df.index], rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Number of adhering frames")
    ax.set_title("Language of generated frames", pad=20)
    ax.legend(title="Detected language")
    return _finish(fig, ax, out)


def plot_grounding(df: pd.DataFrame, out: Path):
    _style()
    df = df.assign(model=df["generator"].map(_short))
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=df, x="model", y="grounding", hue="model", palette="viridis", width=0.6, legend=False, ax=ax)
    median = df["grounding"].median()
    ax.axhline(median, ls="--", color="red", lw=1, label=f"Overall median ({median:.2f})")
    ax.set_title("Factual grounding by model", pad=20)
    ax.set_ylabel("Word overlap with retrieved law text (Jaccard)")
    ax.set_xlabel("Generative model")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    return _finish(fig, ax, out)


def plot_embedder_overlap(scores: pd.Series, out: Path):
    _style()
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(scores.index, scores.values, color=sns.color_palette("viridis", len(scores)))
    ax.bar_label(bars, fmt="%.3f", fontsize=12, padding=3)
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval overlap between embedders per test case", pad=20)
    ax.set_ylabel("Jaccard similarity")
    ax.set_xlabel("Test case")
    return _finish(fig, ax, out)


def plot_retrieval_scores(df: pd.DataFrame, out: Path):
    _style()
    summary = df.groupby("embedder")["top_1_score"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(summary["embedder"].map(lambda m: m.split("/")[-1]), summary["mean"], yerr=summary["std"],
                  capsize=5, color=sns.color_palette("viridis", len(summary)), alpha=0.9)
    ax.bar_label(bars, fmt="%.4f", fontsize=12, padding=5)
    ax.set_ylim(0, summary["mean"].max() * 1.25)
    ax.set_title("Mean top-1 retrieval score by embedding model", pad=20)
    ax.set_ylabel("Mean top-1 retrieval score")
    ax.set_xlabel("Embedding model")
    return _finish(fig, ax, out)


def plot_frames_by_model(df: pd.DataFrame, out: Path):
    _style()
    summary = df.groupby(["embedder", "generator"])["num_frames"].sum().reset_index()
    summary["Embedding model"] = summary["embedder"].map(lambda m: m.split("/")[-1])
    summary["model"] = summary["generator"].map(_short)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=summary, x="model", y="num_frames", hue="Embedding model", palette="viridis", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fontsize=11, padding=3)
    ax.set_title("Unique frames generated per model and embedder", pad=20)
    ax.set_ylabel("Total unique frames")
    ax.set_xlabel("Generative model")
    ax.tick_params(axis="x", rotation=30)
    return _finish(fig, ax, out)


def plot_frame_distribution(df: pd.DataFrame, out: Path):
    _style()
    df = df.assign(model=df["generator"].map(_short),
                   **{"Embedding model": df["embedder"].map(lambda m: m.split("/")[-1])})
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=df, x="model", y="num_frames", hue="Embedding model", palette="viridis", ax=ax)
    ax.set_title("Unique frames per test case", pad=20)
    ax.set_ylabel("Unique frames per test case")
    ax.set_xlabel("Generative model")
    ax.tick_params(axis="x", rotation=30)
    return _finish(fig, ax, out)
