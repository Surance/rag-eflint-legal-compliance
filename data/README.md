# Data

| File | Contents |
|---|---|
| `ISDE_4.5.txt` | Text of the Dutch ISDE regulation (*Investeringssubsidie duurzame energie en energiebesparing*). **Add this file yourself**, see below. |
| `ground_truth.json` | For each test case, the IDs of the relevant chunks, annotated by hand. Used for Recall@k. |

## The law text

The pipeline expects the plain text of the ISDE regulation at `data/ISDE_4.5.txt`. The text comes from the official publication on [wetten.overheid.nl](https://wetten.overheid.nl). Every article must start on its own line as `Artikel x.y.z`, because the chunker splits on that pattern.

Point the pipeline to another file with `--isde-path` (command line) or `isde_path=` (Python).

## Chunk IDs

`chunk_NNN` is the position of a chunk in the list returned by `rag_flint.chunking.load_chunks`. Run `python scripts/export_chunks_for_annotation.py` to see every chunk with its ID.
