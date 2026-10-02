"""Write the cleaned law chunks to a readable text file for manual annotation.

This is how the ground truth in ``data/ground_truth.json`` was made: every
cleaned chunk gets an ID (``chunk_000``, ``chunk_001``, ...) and for each test
case the relevant chunk IDs were selected by hand.

Usage::

    python scripts/export_chunks_for_annotation.py --isde-path data/ISDE_4.5.txt
"""

import argparse
from pathlib import Path

from rag_flint.chunking import load_chunks
from rag_flint.config import DEFAULT_ISDE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--isde-path", type=Path, default=DEFAULT_ISDE_PATH)
    parser.add_argument("--output", type=Path, default=Path("chunks_for_annotation.txt"))
    args = parser.parse_args()

    chunks = load_chunks(args.isde_path.read_text(encoding="utf-8"))
    lines = [
        "=" * 80,
        "Cleaned legal chunks for ground-truth annotation",
        "=" * 80,
        "For each test case (TC01-TC04), list the IDs of the 5-10 most relevant chunks.",
        "",
    ]
    for i, chunk in enumerate(chunks):
        lines += [f"---[ Chunk ID: chunk_{i:03d} ]---", f"Article: {chunk['article']}",
                  f'Text: "{chunk["text"]}"', "-" * 50, ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
