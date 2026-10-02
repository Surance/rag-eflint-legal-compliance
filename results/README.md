# Results

| Path | Contents |
|---|---|
| `summary/frame_quality.csv` | Verbosity, keyword match, concreteness, article references, field fill rate and run time per embedder × generator |
| `summary/frames_per_model.csv` | Number of unique frames per embedder × generator |
| `summary/retrieval.csv` | Mean and standard deviation of the top-1 retrieval score per embedder |
| `summary/embedder_overlap.csv` | Jaccard overlap of the chunks retrieved by the two embedders, per test case |
| `retrieved_chunk_ids.json` | Top-10 chunk IDs per embedder and test case (input for Recall@k) |
| `raw/` | Full output of `python -m rag_flint.experiment`: one JSON file per embedder × generator, with the retrieved chunks, every raw model output and the final frames |

Recompute the summary tables and figures with `python scripts/evaluate.py`.
