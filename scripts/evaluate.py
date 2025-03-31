# Full command:
# python scripts/evaluate.py \
#     --predicted_outputs outputs/phi3_mini/phi3_mini_results_step_500.csv \
#     --gold_outputs outputs/test_gold_id_ood.csv \
#     --word_frequencies outputs/word_frequencies_paisa.json \
#     --word_frequencies_fp_train eureka-rebus/word_frequencies_fp_train.json \
#     --word_frequencies_solution_train eureka-rebus/word_frequencies_solution_train.json \
#     --do_corrs


import json
import argparse
import pandas as pd
from scipy import stats
import seaborn as sns
import jiwer
import matplotlib.pyplot as plt


def get_value_or_empty(list: list, item_idx: int):
    if len(list) > item_idx:
        return list[item_idx]
    return ""


def get_cer(fp: str, solution: str):
    x = [t.lower().replace(" ", "") if isinstance(t, str) else "x" for i, t in enumerate(fp) if list(fp)[i]]
    y = [t.lower().replace(" ", "") if isinstance(t, str) else "x" for i, t in enumerate(solution) if list(fp)[i]]
    out = jiwer.process_characters(x, y)
    return out.cer


def get_correlations(
    words_gold: list[str],
    words_pred: list[str],
    word_freq: pd.DataFrame = None,
    train_freq=None,
    show_corr=True,
    save=False,
):
    # Get per-word accuracy
    words_gold_set = {k: {} for k in set(words_gold)}
    for word in words_gold_set:
        # For each occurrence of the word in the gold set, get the corresponding prediction
        word_preds = [int(g.lower() == p.lower()) for g, p in zip(words_gold, words_pred) if g.lower() == word.lower()]
        words_gold_set[word]["accuracy"] = sum(word_preds) / len(word_preds)
        words_gold_set[word]["num_char"] = len(word)
        if word_freq is not None:
            words_gold_set[word]["freq_ita"] = word_freq[word.lower()] if word.lower() in word_freq else 0
        if train_freq is not None:
            words_gold_set[word]["freq_train"] = train_freq[word.lower()] if word.lower() in train_freq else 0
        
    corr_df = pd.DataFrame(words_gold_set).T
    
    if show_corr:
        plt.figure(figsize=(3, 2))
        sns.heatmap(corr_df.corr(), cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Matrix')
        plt.show()
    
    if save:
        corr_df.to_csv('corr_df.csv', index=False)

    sig_level = 0.01
    corr_fn = lambda x, y: stats.spearmanr(x, y)

    num_char_corr, num_char_p_value = corr_fn(corr_df["accuracy"], corr_df["num_char"])
    print("Word Length correlation:", num_char_corr, "(p-value:", num_char_p_value, "*" if num_char_p_value < sig_level else "", ")")
    if word_freq is not None:
        freq_ita_corr, freq_ita_p_value = corr_fn(corr_df["accuracy"], corr_df["freq_ita"])
        print("Italian Frequency correlation:", freq_ita_corr, "(p-value:", freq_ita_p_value, "*" if freq_ita_p_value < sig_level else "", ")")
    if train_freq is not None:
        freq_train_corr, freq_train_p_value = corr_fn(corr_df["accuracy"], corr_df["freq_train"])
        print("Train Frequency correlation:", freq_train_corr, "(p-value:", freq_train_p_value, "*" if freq_train_p_value < sig_level else "", ")")


def compute_scores(args: argparse.Namespace):
    df_gold = pd.read_csv(args.gold_outputs)
    df_pred = pd.read_csv(args.predicted_outputs)
    word_frequencies_fp_train = json.load(open(args.word_frequencies_fp_train))
    word_frequencies_solution_train = json.load(open(args.word_frequencies_solution_train))

    fp_words_gold = list(x.split(";") if isinstance(x, str) else [] for x in df_gold["word_guesses"])
    fp_words_pred = list(x.split(";") if isinstance(x, str) else [] for x in df_pred["word_guesses"])
    # Flatten
    fp_words_pred = [get_value_or_empty(fp_words_pred[entry_idx], item_idx) for entry_idx, entry in enumerate(fp_words_gold) for item_idx, _ in enumerate(entry)]
    fp_words_gold = [item for sublist in fp_words_gold for item in sublist]
    assert len(fp_words_gold) == len(fp_words_pred)
    accuracies = [int(g.lower() == p.lower()) for g, p in zip(fp_words_gold, fp_words_pred)]

    print("Word Guess Accuracy:" , round(sum(accuracies) / len(accuracies), 2))
    print("Word Guess Length:" , round(sum([int(len(g) == len(p)) for g, p in zip(fp_words_gold, fp_words_pred)]) / len(accuracies), 2))

    tot_fp_accuracy = 0
    fp_letter_accuracy = []
    fp_word_accuracy = []
    fp_word_length = []
    fp_id_word_accuracy = []
    fp_ood_word_accuracy = []
    for i, row in df_gold.iterrows():
        fpg = row["first_pass"]
        fpp = df_pred.loc[i, "first_pass"]
        try:
            tot_fp_accuracy += int(fpg.lower() == fpp.lower()) if isinstance(fpp, str) else 0
        except AttributeError:
            print("Error:", fpg, fpp)
            continue
        fpg_toks = fpg.split() if isinstance(fpg, str) else []
        fpp_toks = fpp.split() if isinstance(fpp, str) else []
        # Check if the number of tokens is the same, pad if not
        if len(fpg_toks) > len(fpp_toks):
            fpp_toks += [""] * (len(fpg_toks) - len(fpp_toks))
        for idx, tok in enumerate(fpg_toks):
            if tok.isupper():
                fp_letter_accuracy.append(int(tok == fpp_toks[idx]))
            else:
                val = int(tok.lower() == fpp_toks[idx].lower())
                fp_word_accuracy.append(val)
                fp_word_length.append(int(len(tok) == len(fpp_toks[idx])))
                if tok.lower() in word_frequencies_fp_train:
                    fp_id_word_accuracy.append(val)
                else:
                    fp_ood_word_accuracy.append(val) 

    print("First Pass Word Accuracy:", round(sum(fp_word_accuracy) / len(fp_word_accuracy),2))
    print("First Pass Word Length:", round(sum(fp_word_length) / len(fp_word_length),2))
    print("First Pass Letter Accuracy:", round(sum(fp_letter_accuracy) / len(fp_letter_accuracy),2))
    print("First Pass Exact Match::", round(tot_fp_accuracy / len(df_gold), 2))
    print("First Pass ID Word Accuracy:", round(sum(fp_id_word_accuracy) / len(fp_id_word_accuracy),2) if len(fp_id_word_accuracy) > 0 else "N/A")
    print("First Pass OOD Word Accuracy:", round(sum(fp_ood_word_accuracy) / len(fp_ood_word_accuracy),2) if len(fp_ood_word_accuracy) > 0 else "N/A")

    solution_words_gold = list(x.split(";") for x in df_gold["solution_words"])
    solution_words_pred = list(x.split(";") if isinstance(x, str) else ["" for _ in range(len(solution_words_gold[i]))] for i, x in enumerate(df_pred["solution_words"]))

    # Flatten
    solution_words_pred = [get_value_or_empty(solution_words_pred[entry_idx], item_idx) for entry_idx, entry in enumerate(solution_words_gold) for item_idx, _ in enumerate(entry)]
    solution_words_gold = [item for sublist in solution_words_gold for item in sublist]
    assert len(solution_words_gold) == len(solution_words_pred)

    solution_word_acc = []
    solution_word_len = []
    solution_id_word_acc = []
    solution_ood_word_acc = []
    for g, p in zip(solution_words_gold, solution_words_pred):
        match = int(g.lower() == p.lower())
        length_match = int(len(g) == len(p))
        solution_word_len.append(length_match)
        solution_word_acc.append(match)
        if g.lower() in word_frequencies_solution_train:
            solution_id_word_acc.append(match)
        else:
            
            solution_ood_word_acc.append(match)

    cer_solution_fp = get_cer(df_pred["first_pass"], df_pred["solution"])
    print("Solution Word Accuracy:" , round(sum(solution_word_acc) / len(solution_word_acc),2))
    print("Solution Word Lengths:" , round(sum(solution_word_len) / len(solution_word_len),2))
    print("Solution First Pass Match:", round(1 - cer_solution_fp))
    print("Solution ID Word Accuracy:" , round(sum(solution_id_word_acc) / len(solution_id_word_acc),2) if len(solution_id_word_acc) > 0 else "N/A")
    print("Solution OOD Word Accuracy:" , round(sum(solution_ood_word_acc) / len(solution_ood_word_acc),2) if len(solution_ood_word_acc) > 0 else "N/A")
    print("Solution Exact Match:" , round(sum([int(g.lower() == p.lower()) if isinstance(p, str) else 0 for g, p in zip(df_gold["solution"], df_pred["solution"])]) / len(df_gold), 2))

    if args.do_corrs:
        word_frequencies = json.load(open(args.word_frequencies))
        print("--- Correlations First Pass ---")
        get_correlations(fp_words_gold, fp_words_pred, word_freq=word_frequencies, train_freq=word_frequencies_fp_train, show_corr=True, save=False)
        print("--- Correlations Solution ---")
        get_correlations(solution_words_gold, solution_words_pred, word_freq=word_frequencies, train_freq=word_frequencies_solution_train, show_corr=True, save=False)
        print("===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_outputs", type=str, required=True)
    parser.add_argument("--gold_outputs", type=str, default="outputs/test_gold_id_ood.csv")
    parser.add_argument("--word_frequencies", type=str, default="outputs/word_frequencies_paisa.json")
    parser.add_argument("--word_frequencies_fp_train", type=str, default="eureka-rebus/word_frequencies_fp_train.json")
    parser.add_argument("--word_frequencies_solution_train", type=str, default="eureka-rebus/word_frequencies_solution_train.json")
    parser.add_argument("--do_corrs", action="store_true", help="Compute correlations")
    args = parser.parse_args()
    compute_scores(args)