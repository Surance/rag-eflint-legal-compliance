"""Recompute all summary tables and figures from the raw experiment results.

Usage::

    python scripts/evaluate.py --results-dir results/raw

Recall@k only needs the files in ``data/`` and ``results/`` that are already in
the repository; the other metrics need the raw JSON output of
``python -m rag_flint.experiment``.
"""

import argparse
import json
from pathlib import Path

from rag_flint.evaluation import metrics, plots

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--summary-dir", type=Path, default=ROOT / "results" / "summary")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    # Recall@k against the hand-annotated ground truth.
    ground_truth = json.loads((ROOT / "data" / "ground_truth.json").read_text())["ground_truth"]
    retrieved = json.loads((ROOT / "results" / "retrieved_chunk_ids.json").read_text())["retrieved"]
    recall = metrics.recall_table(retrieved, ground_truth)
    recall.to_csv(args.summary_dir / "recall_per_test_case.csv", index=False)
    avg_recall = recall.groupby("Embedder")[["Recall@3", "Recall@5", "Recall@10"]].mean()
    avg_recall.to_csv(args.summary_dir / "recall_average.csv")
    print("Average Recall@k\n", avg_recall.round(3), "\n")

    files = sorted(args.results_dir.glob("*.json"))
    if not files:
        print(f"No raw results in {args.results_dir}: skipping frame metrics and figures.")
        return
    records = metrics.load_results(files)
    print(f"Loaded {len(records)} test-case results from {len(files)} files.\n")

    quality = metrics.frame_quality(records)
    quality.to_csv(args.summary_dir / "frame_quality.csv", index=False)
    print(quality.to_string(index=False), "\n")

    counts = metrics.frame_counts(records)
    top1 = metrics.top1_retrieval_scores(records)
    overlap = metrics.embedder_overlap(records)
    top1.groupby("embedder")["top_1_score"].agg(["mean", "std"]).to_csv(args.summary_dir / "retrieval.csv")
    overlap.rename("jaccard_overlap").to_csv(args.summary_dir / "embedder_overlap.csv")

    fig = args.figures_dir
    plots.plot_schema_adherence(metrics.schema_adherence(records), fig / "schema_adherence.png")
    plots.plot_language_distribution(metrics.language_distribution(records), fig / "language_distribution.png")
    plots.plot_grounding(metrics.grounding_scores(records), fig / "factual_grounding_by_model.png")
    plots.plot_embedder_overlap(overlap, fig / "embedder_retrieval_overlap.png")
    plots.plot_retrieval_scores(top1, fig / "retrieval_score_by_embedder.png")
    plots.plot_frames_by_model(counts, fig / "frames_by_embedder_and_model.png")
    plots.plot_frame_distribution(counts, fig / "frame_distribution_boxplot.png")
    print(f"Tables written to {args.summary_dir}, figures to {fig}.")


if __name__ == "__main__":
    main()
