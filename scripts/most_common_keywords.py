# Usage:
# python scripts/most_common_keywords.py \
#   --verbalized_rebus_path "eureka-rebus/verbalized_rebus.csv" \
#   --output_path "eureka-rebus/keyword_frequencies.csv"

import argparse
import unicodedata
from collections import Counter

import pandas as pd


def normalize_keyword(word: str) -> str:
    """Normalize a keyword by stripping accents and non-alphanumeric characters for matching."""
    normalized = unicodedata.normalize('NFD', word)
    normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    normalized = ''.join(c for c in normalized if c.isalnum())
    return normalized.lower()


def main(args: argparse.Namespace):
    print(f"Loading verbalized rebus data from {args.verbalized_rebus_path}...")
    df = pd.read_csv(args.verbalized_rebus_path, escapechar="\\")
    df = df.dropna(subset=["WORDS_PRIMALET"])
    print(f"Loaded {len(df)} verbalized rebuses.")

    # Count keyword frequencies across all rebuses
    keyword_counts = Counter()
    for _, row in df.iterrows():
        for word in row["WORDS_PRIMALET"].split():
            keyword_counts[normalize_keyword(word)] += 1

    # Build output dataframe
    out_df = pd.DataFrame(
        keyword_counts.items(),
        columns=["keyword", "rebus_count"],
    )
    out_df = out_df.sort_values(by="rebus_count", ascending=False).reset_index(drop=True)

    out_df.to_csv(args.output_path, index=False)
    print(f"Found {len(out_df)} unique keywords. Saved to {args.output_path}")

    # Print top 20 for quick inspection
    print(f"\nTop 20 most common keywords:")
    print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbalized_rebus_path", type=str, default="eureka-rebus/verbalized_rebus.csv")
    parser.add_argument("--output_path", type=str, default="eureka-rebus/keyword_frequencies.csv")
    args = parser.parse_args()
    main(args)
