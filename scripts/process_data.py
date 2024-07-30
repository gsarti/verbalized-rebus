# Full command:
# python scripts/process_data.py \
#   --data_folder "eureka-rebus" \
#   --rebus_data_path "rebus.csv" \
#   --crossword_dataset_name "Kamyar-zeinalipour/ITA_CW" \
#   --print_stats \
#   --infer_punctuation \
#   --generate_filtered_rebuses \
#   --filtered_rebus_output_path "rebus_cw_filtered.csv" \
#   --create_train_test_sets \
#   --save_word_frequencies_train \
#   --num_test_examples 1000 \
#   --word_frequencies_first_pass_output_path "word_frequencies_fp_train.csv" \
#   --word_frequencies_solution_output_path "word_frequencies_solution_train.csv" \
#   --save_sharegpt_files

import argparse
import re
import string
import json

import pandas as pd
import numpy as np

from random import sample, seed
from pathlib import Path
from collections import Counter
from datasets import load_dataset


USER_TEMPLATE = "Risolvi gli indizi tra parentesi per ottenere una prima lettura, e usa la chiave di lettura per ottenere la soluzione del rebus.\n\nRebus: {rebus}\nChiave di lettura: {key}"

ANSWER_TEMPLATE = """Procediamo alla risoluzione del rebus passo per passo:
{definitions_solution}

Prima lettura: {first_pass}

Ora componiamo la soluzione seguendo la chiave risolutiva:
{word_by_word_solution}

Soluzione: {solution}
"""


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


def get_stats(df):
    print("# examples", len(df))
    print("# authors", len([x for x in df["AUTORE"].unique() if x != "-" and x is not None]))
    print("Year range", df["ANNO"].min(), df["ANNO"].max())
    fp_word_list = ' '.join([x for x in df['WORDS'] if isinstance(x, str)]).split()
    fp_word_counts = Counter(fp_word_list)
    fp_word_lengths = [len(x.split()) for x in df['WORDS'] if isinstance(x, str)]
    print("Unique FP words", len(fp_word_counts))
    print("Avg. FP num words", sum(fp_word_lengths) / len(df))
    print("SD FP num words", np.std(fp_word_lengths))
    print("Avg. FP word length", sum(len(w) for w in fp_word_counts) / len(fp_word_counts))
    print("SD FP word length", np.std([len(w) for w in fp_word_counts]))
    valid_fp = [len(x) for x in df['PRIMALET'] if isinstance(x, str)]
    print("Avg. FP length", sum(valid_fp) / len(valid_fp))
    print("SD FP length", np.std(valid_fp))
    print("===")
    s_word_list = [w for x in df['FRASE_SEPARATED'] for w in x.split() if w not in string.punctuation]
    s_word_counts = Counter(s_word_list)
    s_word_lengths = [len([w for w in x.split() if w not in string.punctuation]) for x in df['FRASE_SEPARATED']]
    print("Unique solution words", len(s_word_counts))
    print("Avg. solution num words", sum(s_word_lengths) / len(df))
    print("SD solution num words", np.std(s_word_lengths))
    print("Avg. solution word length", sum(len(w) for w in s_word_counts) / len(s_word_counts))
    print("SD solution word length", np.std([len(w) for w in s_word_counts]))
    valid_s = [len(x) for x in df['FRASE'] if isinstance(x, str)]
    print("Avg. solution length", sum(valid_s) / len(valid_s))
    print("SD solution length", np.std(valid_s))


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


def get_sharegpt_message(row):
    rebus = row["VERBALIZED_PRIMALET"]
    words = row["WORDS"]
    key = row["FRASE_LEN"]
    first_pass = row["PRIMALET"]
    solution_sep = row["FRASE_SEPARATED"]
    solution = row["FRASE"]
    if not isinstance(rebus, str) or not isinstance(words, str) or not isinstance(key, str) or not isinstance(first_pass, str) or not isinstance(solution, str) or not isinstance(solution_sep, str):
        return None
    try:
        message = [
            {
                'from': 'human', 'value': USER_TEMPLATE.format(rebus=rebus, key=key)
            },
            {
                'from': 'gpt', 'value': ANSWER_TEMPLATE.format(
                    key=key,
                    definitions_solution=get_definitions_solution(rebus, words),
                    first_pass=get_firstpass(first_pass),
                    word_by_word_solution=get_word_by_word_solution(key, solution_sep),
                    solution=solution
                )
            }
        ]
    except IndexError:
        return None
    return message


def process_rebus_data(args: argparse.Namespace):
    args.rebus_data_path = Path(args.data_folder) / args.rebus_data_path
    args.filtered_rebus_output_path = Path(args.data_folder) / args.filtered_rebus_output_path
    args.word_frequencies_first_pass_output_path = Path(args.data_folder) / args.word_frequencies_first_pass_output_path
    args.word_frequencies_solution_output_path = Path(args.data_folder) / args.word_frequencies_solution_output_path
    args.ood_words_output_path = Path(args.data_folder) / args.ood_words_output_path

    print(f"Loading rebus data from {args.rebus_data_path}...")
    df = pd.read_csv(args.rebus_data_path, escapechar="\\")

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
    if args.infer_punctuation:
        punctuation = set()
        for s in df["FRASE"]:
            punctuation.update(set(c for c in s if not c.isalnum() and not c.isspace()))
    else:
        punctuation = string.punctuation
    
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

    if args.print_stats:
        print("=" * 10 + " EurekaRebus Stats " + "=" * 10)
        get_stats(df)
        print("=" * 20)

    if args.generate_filtered_rebuses:
        print(f"Loading crossword dataset {args.crossword_dataset_name}...")
        crosswords = load_dataset(args.crossword_dataset_name)["train"].to_pandas()
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
        filtered_df.to_csv(args.filtered_rebus_output_path, index=False)
        print(f"Filtered rebuses saved to {args.filtered_rebus_output_path}.")

        if args.print_stats:
            print("=" * 10 + " Filtered EurekaRebus Stats " + "=" * 10)
            get_stats(filtered_df)
            print("=" * 20)
    else:
        filtered_df = df
    
    if args.create_train_test_sets:
        train_df, in_domain_test_df, ood_test_df = create_test_sets(
            filtered_df,
            min_freq_word=args.ood_min_freq_word,
            max_freq_word=args.ood_max_freq_word,
            num_examples=args.num_test_examples,
            save_ood_words_path=args.ood_words_output_path
        )
        print(f"Train set: {len(train_df)} examples")
        print(f"In-domain test set: {len(in_domain_test_df)} examples")
        print(f"Out-of-domain test set: {len(ood_test_df)} examples")

        if args.save_word_frequencies_train:
            print("Saving word frequencies for train set...")
            # Split the words and count their occurrences
            word_list_fp = ' '.join([x for x in train_df['WORDS'] if isinstance(x, str)]).split()
            word_counts_fp = Counter(word_list_fp)
            word_counts_fp_df = pd.DataFrame(word_counts_fp.items(), columns=['Word', 'Count'])
            word_counts_fp_df = word_counts_fp_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
            word_counts_fp_dic = word_counts_fp_df.set_index(word_counts_fp_df.columns[0])[word_counts_fp_df.columns[1]].to_dict()
            with open(args.word_frequencies_first_pass_output_path, 'w') as f:
                json.dump(word_counts_fp_dic, f)
            
            word_list_solution = [w for x in train_df['FRASE_SEPARATED'] for w in x.split() if w not in string.punctuation]
            word_counts_solution = Counter(word_list_solution)
            word_counts_solution_df = pd.DataFrame(word_counts_solution.items(), columns=['Word', 'Count'])
            word_counts_solution_df = word_counts_solution_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
            word_counts_solution_dic = word_counts_solution_df.set_index(word_counts_solution_df.columns[0])[word_counts_solution_df.columns[1]].to_dict()
            with open(args.word_frequencies_solution_output_path, 'w') as f:
                json.dump(word_counts_solution_dic, f)
            print(f"Word frequencies for train set saved to {args.data_folder}.")

        datasets = {
            "train": train_df,
            "id_test": in_domain_test_df,
            "ood_test": ood_test_df
        }
    else:
        datasets = {
            "all": filtered_df
        }
    
    if args.save_sharegpt_files:
        for name, df in datasets.items():
            sharegpt_texts = []
            for _, row in df.iterrows():
                out = get_sharegpt_message(row)
                if out is not None:
                    sharegpt_texts.append(out)
            with open(Path(args.data_folder) / f"{name}.jsonl", 'w') as f:
                for text in sharegpt_texts:
                    jsontxt = json.dumps({"conversations": text})
                    f.write(jsontxt + "\n")
            print(f"ShareGPT files for {name} verbalized rebuses set saved to {args.data_folder}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="eureka-rebus")
    parser.add_argument("--rebus_data_path", type=str, default="rebus.csv")
    parser.add_argument("--crossword_dataset_name", type=str, default="Kamyar-zeinalipour/ITA_CW")
    parser.add_argument("--print_stats", action="store_true")
    parser.add_argument("--infer_punctuation", action="store_true")
    parser.add_argument("--generate_filtered_rebuses", action="store_true")
    parser.add_argument("--filtered_rebus_output_path", type=str, default="verbalized_rebus.csv")
    parser.add_argument("--create_train_test_sets", action="store_true")
    parser.add_argument("--save_word_frequencies_train", action="store_true")
    parser.add_argument("--num_test_examples", type=int, default=1000)
    parser.add_argument("--ood_min_freq_word", type=int, default=10)
    parser.add_argument("--ood_max_freq_word", type=int, default=15)
    parser.add_argument("--ood_words_output_path", type=str, default="ood_words.txt")
    parser.add_argument("--word_frequencies_first_pass_output_path", type=str, default="word_frequencies_fp_train.json")
    parser.add_argument("--word_frequencies_solution_output_path", type=str, default="word_frequencies_solution_train.json")
    parser.add_argument("--save_sharegpt_files", action="store_true")
    args = parser.parse_args()
    process_rebus_data(args)