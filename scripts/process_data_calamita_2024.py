# Full command:
# python scripts/process_data_calamita_2024.py \
#   --data_folder "eureka-rebus-calamita-2024" \
#   --rebus_data_path "eureka-rebus/rebus.csv" \
#   --crossword_dataset_name "Kamyar-zeinalipour/ITA_CW" \
#   --infer_punctuation \
#   --generate_filtered_rebuses \
#   --filtered_rebus_output_path "rebus_cw_filtered.csv" \
#   --create_train_test_sets \
#   --num_test_examples 1000

import re
import string

import pandas as pd

from random import sample, seed
from pathlib import Path
from collections import Counter
from datasets import load_dataset

DATA_FOLDER = "eureka-rebus-calamita-2024"
REBUS_DATA_PATH = "eureka-rebus/rebus.csv"
CROSSWORD_DATASET_NAME = "Kamyar-zeinalipour/ITA_CW"
NUM_TEST_EXAMPLES = 1000

def get_words_letters_from_first_pass(first_pass: str) -> tuple[str, str]:
    curr_words = []
    curr_letters = []
    if isinstance(first_pass, str):
        for seg in first_pass.split("-"):
            if isinstance(seg, str):
                for w in seg.strip().split():
                    # Possibly revise this condition to account for words like "D'Annunzio"
                    if isinstance(w, str) and (w.islower() or (len(w) > 1 and (w.istitle() or w[0].islower() and w[-1].isupper()))):
                        curr_words.append(w)
                    elif isinstance(w, str) and w.isupper():
                        for l in w:
                            curr_letters.append(l)
    return " ".join(curr_words), " ".join(curr_letters)

def get_solution_key_and_sep(solution: str, punctuation: str = string.punctuation) -> tuple[str, str]:
    curr_frase_len = []
    for p in punctuation:
        solution = solution.replace(p, f" {p} ")
    words = solution.split()
    for word in words:
        if word in punctuation:
            curr_frase_len.append(word)
        else:
            curr_frase_len.append(str(len(word)))
    return " ".join(curr_frase_len), " ".join(words)


def build_verbalized_rebus(first_pass: str, words: list[str], matches: dict[str, str]) -> str:
    # Randomly sample a clue for each word among available clues
    replacements = [sample(matches[word.lower()], 1)[0] for word in words]

    # Done to avoid accidentally replacing words from previous replacements
    for idx, word in enumerate(words):
        first_pass = first_pass.replace(word, f"[{idx}]")
    
    # Replace words with their respective replacements
    verbalized = first_pass
    verbalized_with_len = first_pass
    for idx, (word, replacement) in enumerate(zip(words, replacements)):
        verbalized = verbalized.replace(f"[{idx}]", f"[{replacement}]")
        verbalized = re.sub(' +', ' ', verbalized.replace("-", ""))
        verbalized_with_len = verbalized_with_len.replace(f"[{idx}]", f"[{replacement} ({len(word)})]")
        verbalized_with_len = re.sub(' +', ' ', verbalized_with_len.replace("-", ""))
    return verbalized, verbalized_with_len


def create_test_sets(df, min_freq_word, max_freq_word, num_examples, save_ood_words_path: str = None):
    # Get all unique words from the dataset
    df = df.dropna(subset=['WORDS'])
    word_list = ' '.join([x for x in df['WORDS'] if isinstance(x, str)]).split()
    word_counts = Counter(word_list)

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Assumes all words are in different rows
    ood_count_estimate = 0
    ood_words = []

    train_idx = []
    in_domain_test_idx = []
    ood_test_idx = []

    for idx, row in df.iterrows():
        words = set(row['WORDS'].split())
        if ood_count_estimate < num_examples or any(word in ood_words for word in words):
            for word in words:
                if word_counts[word] < min_freq_word or word_counts[word] > max_freq_word:
                    continue
                if word not in ood_words and ood_count_estimate < num_examples:
                    ood_words.append(word)
                    ood_count_estimate += word_counts[word]
                    
    for idx, row in df.iterrows():    
        words = set(row['WORDS'].split())
        if any(word in ood_words for word in words):
            ood_test_idx.append(idx)
        elif len(in_domain_test_idx) < num_examples:
            in_domain_test_idx.append(idx)
        else:
            train_idx.append(idx)

    # Create dataframes
    train_df = df.iloc[train_idx]
    in_domain_test_df = df.iloc[in_domain_test_idx]
    ood_test_df = df.iloc[ood_test_idx]

    ood_words_loc = []
    for idx, row in ood_test_df.iterrows():
        words = set(row['WORDS'].split())
        curr_locs = []
        for widx, word in enumerate(words):
            if word in ood_words:
                curr_locs.append(str(widx))
        ood_words_loc.append(" ".join(curr_locs))
    ood_test_df['OOD_WORDS_LOC'] = ood_words_loc

    if save_ood_words_path is not None:
        with open(save_ood_words_path, 'w') as f:
            for word in sorted(ood_words):
                f.write(word + "\n")
        print(f"Out-of-domain words saved to {save_ood_words_path}.")

    return train_df, in_domain_test_df, ood_test_df


def get_definitions(rebus):
    return re.findall(r"(\[.*?\])", rebus)


def get_letters_and_definitions(rebus):
    vals = []
    curr_definition = 0
    definitions = get_definitions(rebus)
    for tok in re.sub(' +', ' ', re.sub("\[.*?\]", " <<<>>> ", rebus)).split():
        if tok != "<<<>>>":
            if len(tok) > 1:
                for i in range(len(tok) - 1):
                    vals.append(tok[i])
                curr_val = tok[-1]
            else:
                curr_val = tok
        else:
            curr_val = definitions[curr_definition]
            curr_definition += 1
        vals.append(curr_val)
    return vals


def get_definitions_solution(rebus, words):
    curr_definition = 0
    definitions = get_definitions(rebus)
    words = words.split()
    out = []
    curr_letters = []
    for tok in get_letters_and_definitions(rebus):
        if tok.startswith("["):
            if len(curr_letters) > 1:
                out.append(f"- {' '.join(curr_letters)} = {' '.join(curr_letters)}")
            elif len(curr_letters) == 1:
                out.append(f"- {curr_letters[0]} = {curr_letters[0]}")
            curr_letters = []
            out.append(f"- {definitions[curr_definition]} = {words[curr_definition]}")
            curr_definition += 1
        else:
            curr_letters.append(tok)
    if len(curr_letters) > 1:
        out.append(f"- {' '.join(curr_letters)} = {' '.join(curr_letters)}")
    elif len(curr_letters) == 1:
        out.append(f"- {curr_letters[0]} = {curr_letters[0]}")
    return "\n".join(out)


def get_firstpass(firstpass):
    return re.sub(' +', ' ', firstpass.replace("-", "")).strip()


def get_word_by_word_solution(key, solution):
    return "\n".join(f"{length} = {word}" if length.isnumeric() else f"{word} = {word}" for length, word in zip(key.split(), solution.split()))


def filter_df_calamita(df: pd.DataFrame):
    return pd.DataFrame({
        "verbalized_rebus": df["VERBALIZED_PRIMALET"].to_list(),
        "verbalized_rebus_with_length_hints": df["VERBALIZED_PRIMALET_WITH_LEN"].to_list(),
        "solution_key": df["FRASE_LEN"].to_list(),
        "word_guesses": [";".join(s.split()) for s in df["WORDS"]],
        "first_pass": [re.sub(' +', ' ', fp.replace("-", "")) for fp in df["PRIMALET"]],
        "solution_words": [";".join(s.split()) for s in df["FRASE"]],
        "solution": df["FRASE"].to_list(),
    })

def process_rebus_data():
    rebus_data_path = Path(REBUS_DATA_PATH)
    train_output_path = Path(DATA_FOLDER) / "train.csv"
    test_output_path = Path(DATA_FOLDER) / "test.csv"

    print(f"Loading rebus data from {rebus_data_path}...")
    df = pd.read_csv(rebus_data_path, escapechar="\\")

    print(f"Found {len(df)} unique rebuses. Formatting rebus fields...")
    df['ANNO'] = pd.to_numeric(df['ANNO'])
    df['MESE'] = pd.to_numeric(df['MESE'])
    df["FRASE"] = df["FRASE"].str.strip()

    words = []
    letters = []
    for _, row in df.iterrows():
        curr_words, curr_letters = get_words_letters_from_first_pass(row["PRIMALET"])
        words.append(curr_words)
        letters.append(curr_letters)
    df["WORDS"] = words
    df["LETTERS"] = letters

    # Get all punctuation marks from a set of strings
    punctuation = set()
    for s in df["FRASE"]:
        punctuation.update(set(c for c in s if not c.isalnum() and not c.isspace()))
    
    solution_keys = []
    solution_sep = []
    for _, row in df.iterrows():
        curr_solution_key, curr_solution_sep = get_solution_key_and_sep(row["FRASE"], punctuation)
        solution_keys.append(curr_solution_key)
        solution_sep.append(curr_solution_sep)
    df["FRASE_LEN"] = solution_keys
    df["FRASE_SEPARATED"] = solution_sep

    # Deduplicate rows
    df = df.drop_duplicates(subset=["FRASE", "PRIMALET"])

    print(f"Loading crossword dataset {CROSSWORD_DATASET_NAME}...")
    crosswords = load_dataset(CROSSWORD_DATASET_NAME)["train"].to_pandas()
    print(f"Loaded {len(crosswords)} crossword clues. Matching rebuses with crossword clues...")

    matches = {}
    for _, row in crosswords.iterrows():
        if row["Answer"] is not None and row["Clue"] is not None:
            answer, clue = row["Answer"].lower(), row["Clue"]
            if answer not in matches:
                matches[answer] = [clue]
            else:
                matches[answer].append(clue)

    # Filters:
    # 1. Consider only rebus with at least 2 words
    # 2. Consider only regular rebus
    # 3. Consider only rebus for which all words are matched
    filtered_df = df[
        (~df["PRIMALET"].isna()) &
        (len(df["WORDS"].str.split()) >= 2) &
        (df["TIPO"].isna()) &
        (df["WORDS"].apply(lambda x: all(w.lower() in matches for w in x.split())))
    ]
    print(f"Filtered {len(filtered_df)} rebuses matching crossword definitions. Generating encrypted sequences...")

    verbalized_rebus = []
    verbalized_rebus_with_len = []
    seed(42)
    for _, row in filtered_df.iterrows():
        curr_verbalized_rebus, curr_verbalized_rebus_with_len = build_verbalized_rebus(
            row["PRIMALET"], row["WORDS"].split(), matches
        )
        verbalized_rebus.append(curr_verbalized_rebus)
        verbalized_rebus_with_len.append(curr_verbalized_rebus_with_len)

    filtered_df["VERBALIZED_PRIMALET"] = verbalized_rebus
    filtered_df["VERBALIZED_PRIMALET_WITH_LEN"] = verbalized_rebus_with_len
    train_df, in_domain_test_df, ood_test_df = create_test_sets(
        filtered_df,
        min_freq_word=10,
        max_freq_word=15,
        num_examples=NUM_TEST_EXAMPLES,
    )
    test_df = pd.concat([in_domain_test_df, ood_test_df])

    # Added filtering to keep only evaluation fields
    filtered_train_df = filter_df_calamita(train_df)
    filtered_test_df = filter_df_calamita(test_df)
    filtered_train_df.to_csv(train_output_path, index=False)
    filtered_test_df.to_csv(test_output_path, index=False)
    print(f"Train set: {len(filtered_train_df)} examples - Saved to {train_output_path}")
    print(f"Test set: {len(filtered_test_df)} examples - Saved to {test_output_path}")

if __name__ == "__main__":
    process_rebus_data()