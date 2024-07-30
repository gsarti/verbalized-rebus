# Full command:
# python scripts/prompt_models.py \
#   --model gpt \
#   --api_key YOUR_API_KEY

import os
import re
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
from guidance import models, user, assistant, gen
from datasets import load_dataset

regex_word_guess = '[\d\.|-] \[.* = (.*)'
regex_firstpass = 'Prima lettura: (.*)'
regex_solution_word = "\d+ = (.*)"
regex_solution = "Soluzione: (.*)"

PROMPT = """Risolvi gli indizi tra parentesi per ottenere una prima lettura, e usa la chiave di lettura per ottenere la soluzione del rebus.
# Esempio 1:

Rebus: AC [Un mollusco nell'insalata di mare] GLI [Lo è l'operaio che lavora in cantiere] S TO [Soldati da trincea]
Chiave di lettura: 11 2 10

Procediamo alla risoluzione del rebus passo per passo:
- A C = A C
- [Un mollusco nell'insalata di mare] = cozza
- G L I = G L I
- [Lo è l'operaio che lavora in cantiere] = edile
- S T O = S T O
- [Soldati da trincea] = fanti

Prima lettura: AC cozza GLI edile S TO fanti

Ora componiamo la soluzione seguendo la chiave risolutiva:
11 = Accozzaglie
2 = di
10 = lestofanti

Soluzione: Accozzaglie di lestofanti

# Esempio 2:

Rebus: [Edificio religioso] G [Lo fa doppio l'opportunista] NP [Poco cortese, severo] NZ [Parente... molto lontana]
Chiave di lettura: 3 1 6 3 8 2

Procediamo alla risoluzione del rebus passo per passo:
- [Edificio religioso] = chiesa
- G = G
- [Lo fa doppio l'opportunista] = gioco
- N P = N P
- [Poco cortese, severo] = rude
- N Z = N Z
- [Parente... molto lontana] = ava

Prima lettura: chiesa G gioco NP rude NZ ava

Ora componiamo la soluzione seguendo la chiave risolutiva:
3 = Chi
1 = è
6 = saggio
3 = con
8 = prudenza
2 = va

Soluzione: Chi è saggio con prudenza va

# Esempio 3:

Rebus: MO [Venatura del marmo] B [Incitamento nel sollevamento] L I
Chiave di lettura: 6 8

Procediamo alla risoluzione del rebus passo per passo:
- M O = M O
- [Venatura del marmo] = stria
- B = B
- [Incitamento nel sollevamento] = issa
- L I = L I

Prima lettura: MO stria B issa L I

Ora componiamo la soluzione seguendo la chiave risolutiva:
6 = Mostri
8 = abissali

Soluzione: Mostri abissali

# Esempio 4:

Rebus: [Tarda quando è sospirato] N IS [Mezzo di trasporto cittadino] A [Metropoli brasiliana] LE [Malvagia per il poeta]
Chiave di lettura: 8 10

Procediamo alla risoluzione del rebus passo per passo:
- [Tarda quando è sospirato] = si
- N I S = N I S
- [Mezzo di trasporto cittadino] = tram
- A = A
- [Metropoli brasiliana] = rio
- L E = L E
- [Malvagia per il poeta] = ria

Prima lettura: si N IS tram A rio LE ria

Ora componiamo la soluzione seguendo la chiave risolutiva:
8 = Sinistra
10 = marioleria

Soluzione: Sinistra marioleria

# Esempio 5:

Rebus: [Fa sudare i soldati] [La chiave di violino] [L' ...ultimo esponente]
Chiave di lettura: 6 7

Procediamo alla risoluzione del rebus passo per passo:
- [Fa sudare i soldati] = marcia
- [La chiave di violino] = sol
- [L' ...ultimo esponente] = Enne

Prima lettura: marcia sol Enne

Ora componiamo la soluzione seguendo la chiave risolutiva:
6 = Marcia
7 = solenne

Soluzione: Marcia solenne

# Ora tocca a te! Completa il rebus seguendo il procedimento descritto, rispondendo esattamente nello stesso formato utilizzato dagli esempi precedenti.

Rebus: {rebus}
Chiave di lettura: {key}

"""

def get_rebus_and_key(ex):
    rebus_and_key = ex["conversations"][0]["value"][130:].split("\n")
    rebus = rebus_and_key[0].split(": ")[1]
    key = rebus_and_key[1].split(": ")[1]
    return rebus, key

def init_llama_tokenizer(hf_token):
    from transformers import AutoTokenizer
    from tokenizers import pre_tokenizers

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-70B-Instruct",
        padding_side="left",
        trust_remote_code=True,
        token=hf_token,
    )

    byte_decoder = {}
    known_vals = set([])

    for j in range(256):
        for k in range(256):
            for l in range(256):
                if len(byte_decoder.keys()) < 256:
                    b = b""
                    vals = [j,k,l]
                    if not set(vals).issubset(known_vals):
                        for d in range(3):
                            b = b + int.to_bytes(vals[d])
                        try:
                            c = b.decode()
                            t = pre_tokenizers.ByteLevel(False,False).pre_tokenize_str(c)[0][0]
                            for m in range(3):
                                if t[m] not in byte_decoder.keys():
                                    byte_decoder[t[m]] = vals[m]
                                    known_vals.add(vals[m])
                        except UnicodeDecodeError:
                            pass
    byte_decoder['À'] = 192
    byte_decoder['Á'] = 193
    byte_decoder['ð'] = 240
    byte_decoder['ñ'] = 241
    byte_decoder['ò'] = 242
    byte_decoder['ó'] = 243
    byte_decoder['ô'] = 244
    byte_decoder['õ'] = 245
    byte_decoder['ö'] = 246
    byte_decoder['÷'] = 247
    byte_decoder['ø'] = 248
    byte_decoder['ù'] = 249
    byte_decoder['ú'] = 250
    byte_decoder['û'] = 251
    byte_decoder['ü'] = 252
    byte_decoder['ý'] = 253
    byte_decoder['þ'] = 254
    byte_decoder['ÿ'] = 255
    tokenizer.byte_decoder = byte_decoder
    return tokenizer

def parse_generation(ex_idx, lm):
    try:
        word_guesses = ";".join(re.findall(regex_word_guess, lm["output"]))
    except:
        word_guesses = ""
    try:
        first_pass = re.findall(regex_firstpass, lm["output"])[0]
    except:
        first_pass = ""
    try:
        solution_words = ";".join(re.findall(regex_solution_word, lm["output"]))
    except:
        solution_words = ""
    try:
        solution = re.findall(regex_solution, lm["output"])[0]
    except:
        solution = ""
    return {
        "idx": ex_idx,
        "word_guesses": word_guesses,
        "first_pass": first_pass,
        "solution_words": solution_words,
        "solution": solution,
    }

def generate(args: argparse.Namespace):
    eval_dataset = load_dataset('gsarti/eureka-rebus', 'llm_sft', data_files=["id_test.jsonl", "ood_test.jsonl"], split = "train")
    
    # Load model
    if args.model == "claude":
        model = models.Anthropic("claude-3-5-sonnet-20240620", api_key=args.api_key, echo=False)
    elif args.model == "gpt":
        model = models.OpenAI("gpt-4o", api_key=args.api_key, echo=False)
    elif args.model == "llama":
        # Black magic needed to initialize the LLaMA tokenizer with Guidance
        tokenizer = init_llama_tokenizer(args.hf_token)
        model = models.TogetherAIChat("meta-llama/Meta-Llama-3-70B-Instruct", tokenizer=tokenizer, api_key=args.api_key, echo=False)
    elif args.model == "qwen":
        model = models.TogetherAIChat("Qwen/Qwen2-72B-Instruct", api_key=args.api_key, echo=False)
    else:
        raise ValueError("Invalid model")

    results = {}
    outputs = []
    # JSON: Temporary map ID -> parsed results
    if os.path.exists(f'outputs/prompted_models/{args.model}_results.json'):
        results = json.load(open(f'outputs/prompted_models/{args.model}_results.json'))
    # TXT: Full model generation
    if os.path.exists(f'outputs/prompted_models/{args.model}_outputs.txt'):
        outputs = open(f'outputs/prompted_models/{args.model}_outputs.txt').read().split("=" * 10)
    for ex_idx, ex in tqdm(enumerate(eval_dataset), total=len(eval_dataset)):
        if str(ex_idx) in results and results[str(ex_idx)]["word_guesses"] != "":
            continue
        rebus, key = get_rebus_and_key(ex)
        with user():
            lm = model + args.prompt.format(rebus=rebus, key=key)

        with assistant():
            try:
                lm += gen(name="output")
            except Exception as e:
                print(f"Skipped {ex_idx}: {e}")
                results[str(ex_idx)] = {
                    "idx": ex_idx,
                    "word_guesses": "",
                    "first_pass": "",
                    "solution_words": "",
                    "solution": "",
                }
                outputs.append(f"\n\n{ex_idx}: {lm['output']}\n\n")
                json.dump(results, open(f'outputs/prompted_models/{args.model}_results.json', 'w'), indent=4)
                open(f'outputs/prompted_models/{args.model}_outputs.txt', 'w').write(("=" * 10).join(outputs))
                continue

        outputs.append(f"\n\n{ex_idx}: {lm['output']}\n\n")
        results[str(ex_idx)] = parse_generation(ex_idx, lm)
        open(f'outputs/prompted_models/{args.model}_outputs.txt', 'w').write(("=" * 10).join(outputs))
        json.dump(results, open(f'outputs/prompted_models/{args.model}_results.json', 'w'), indent=4)

        # Sleep to avoid rate limiting
        time.sleep(args.sleep_rate_limit)

    # Final conversion to CSV
    final_results = json.load(open(f'outputs/prompted_models/{args.model}_results.json'))
    for i in range(2000):
        if str(i) not in final_results:
            print("Missing", i)
            final_results.append({f"{i}": {
                "idx": i,
                "word_guesses": "",
                "first_pass": "",
                "solution_words": "",
                "solution": ""
            }})
    ordered_results = []
    for i in range(2000):
        ordered_results.append(final_results[str(i)])
    df = pd.DataFrame(ordered_results)
    df.to_csv(f'outputs/prompted_models/{args.model}_results.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["claude", "gpt", "llama", "qwen"])
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--sleep_rate_limit", type=int, default=12)
    args = parser.parse_args()
    if args.prompt is None:
        args.prompt = PROMPT
    if "{rebus}" not in args.prompt or "{key}" not in args.prompt:
        raise ValueError("Prompt must contain {rebus} and {key} placeholders")
    if args.model == "llama" and args.hf_token is None:
        raise ValueError("--hf_token must be provided for LLaMA model")
    generate(args)
    

