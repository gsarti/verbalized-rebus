# Usage:
# python scripts/find_unmatched_keywords.py \
#   --rebus_data_path "eureka-rebus/rebus.csv" \
#   --crossword_dataset_name "cruciverb-it/evalita2026" \
#   --output_path "eureka-rebus/unmatched_keywords.csv"

import argparse
import string
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def normalize_keyword(word: str) -> str:
    """Normalize a keyword by stripping accents and non-alphanumeric characters for matching."""
    normalized = unicodedata.normalize('NFD', word)
    normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    normalized = ''.join(c for c in normalized if c.isalnum())
    return normalized.lower()


def get_words_from_first_pass(first_pass: str) -> list[str]:
    """Extract lowercase/titlecase words from a first pass string."""
    curr_words = []
    if isinstance(first_pass, str):
        for seg in first_pass.split("-"):
            if isinstance(seg, str):
                for w in seg.strip().split():
                    if isinstance(w, str) and (w.islower() or (len(w) > 1 and (w.istitle() or (w[0].islower() and w[-1].isupper())))):
                        curr_words.append(w)
    return curr_words


def main(args: argparse.Namespace):
    print(f"Loading rebus data from {args.rebus_data_path}...")
    df = pd.read_csv(args.rebus_data_path, escapechar="\\")
    df = df.dropna(subset=["PRIMALET"])
    print(f"Loaded {len(df)} rebuses with a first pass.")

    print(f"Loading crossword dataset {args.crossword_dataset_name}...")
    data_files = args.crossword_data_files if args.crossword_data_files else None
    crosswords = load_dataset(args.crossword_dataset_name, data_files=data_files, split="train").to_pandas()
    print(f"Loaded {len(crosswords)} crossword clues.")

    # Build set of known crossword answers
    answers = set()
    for _, row in crosswords.iterrows():
        if row[args.crossword_answer_column] is not None:
            answers.add(normalize_keyword(row[args.crossword_answer_column]))

    # Load extra definitions and add to answers
    extra_defs_path = Path(args.extra_definitions_path)
    if extra_defs_path.exists():
        extra_defs = pd.read_csv(extra_defs_path)
        for _, row in extra_defs.iterrows():
            answers.add(normalize_keyword(str(row["keyword"])))
        print(f"Loaded {len(extra_defs)} extra definitions from {extra_defs_path}.")
    else:
        print(f"Extra definitions file not found at {extra_defs_path}, skipping.")

    # Count unmatched keywords across all rebuses
    unmatched_counts = Counter()
    for _, row in df.iterrows():
        words = get_words_from_first_pass(row["PRIMALET"])
        for word in words:
            if normalize_keyword(word) not in answers:
                unmatched_counts[word.lower()] += 1

    # Build output dataframe
    out_df = pd.DataFrame(
        unmatched_counts.items(),
        columns=["keyword", "rebus_count"],
    )
    out_df = out_df.sort_values(by="rebus_count", ascending=False).reset_index(drop=True)

    out_df.to_csv(args.output_path, index=False)
    print(f"Found {len(out_df)} unmatched keywords. Saved to {args.output_path}")

    # Print top 20 for quick inspection
    print(f"\nTop 20 unmatched keywords:")
    print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebus_data_path", type=str, default="eureka-rebus/rebus.csv")
    parser.add_argument("--crossword_dataset_name", type=str, default="cruciverb-it/evalita2026")
    parser.add_argument("--crossword_data_files", type=str, default="task_1/datasets/train.csv")
    parser.add_argument("--crossword_answer_column", type=str, default="answer")
    parser.add_argument("--crossword_clue_column", type=str, default="clue")
    parser.add_argument("--extra_definitions_path", type=str, default="eureka-rebus/missing_keywords_extra_definitions.csv")
    parser.add_argument("--output_path", type=str, default="eureka-rebus/unmatched_keywords.csv")
    args = parser.parse_args()
    main(args)
