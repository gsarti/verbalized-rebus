# Non Verbis, Sed Rebus: Large Language Models are Weak Solvers of Italian Rebuses

[Gabriele Sarti](https://gsarti.com) • [Tommaso Caselli](https://scholar.google.com/citations?user=fxQvP_QAAAAJ) • [Malvina Nissim](https://malvinanissim.github.io/) • [Arianna Bisazza](https://www.cs.rug.nl/~bisazza/)

<p float="left">
    <img src="img/verbalized_rebus.png" alt="An example of a verbalized rebus" width="300"/>
    <img src="img/llm_generations.png" alt="Example generations from LLMs" width="300"/>
</p>

> **Abstract:** Rebuses are puzzles requiring constrained multi-step reasoning to identify a hidden phrase from a set of images and letters. In this work, we introduce a large collection of verbalized rebuses for the Italian language and use it to assess the rebus-solving capabilities of state-of-the-art large language models. While general-purpose systems such as LLaMA-3 and GPT-4o perform poorly on this task, ad-hoc fine-tuning seems to improve models' performance. However, we find that performance gains from training are largely motivated by memorization. Our results suggest that rebus solving remains a challenging test bed to evaluate large language models' linguistic proficiency and sequential instruction-following skills.

This repository contains scripts and notebooks associated with the paper ["Non Verbis, Sed Rebus: Large Language Models are Weak Solvers of Italian Rebuses"](TBD). If you use any of the following contents for your work, we kindly ask you to cite our paper:

```bibtex
TBD
```

All models and data used in this work are available in our [🤗 Hub Collection](TBD).

## Try it yourself! 🧩

We provide a simple online demo to test the rebus-solving capabilities of our model. You can access it [here](TBD).

## Installation

To install the required dependencies, you can use the following command:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run [training and inference notebooks](./notebooks) you will need a machine with access to a GPU (required by [Unsloth](https://github.com/unslothai/unsloth)). The environment setup is performed in the first cell of the training notebooks.

## Reproducing the paper results

### Data preprocessing

> ⚠️ Refer to the [EurekaRebus]([eureka-rebus/README.md](https://huggingface.co/datasets/gsarti/eureka-rebus)) dataset card for more information on the dataset and the [data licensing](https://huggingface.co/datasets/gsarti/eureka-rebus/raw/main/LICENSE).

Run the following command to produce all data contained in the `eureka-rebus` folder from the `rebus.csv` file:

```shell
python scripts/process_data.py \
  --print_stats \
  --infer_punctuation \
  --generate_filtered_rebuses \
  --create_train_test_sets \
  --save_word_frequencies_train \
  --save_sharegpt_files
```

### Fine-tuning LLMs on EurekaRebus

Follow the notebooks [train_phi3_mini](notebooks/train_phi3_mini.ipynb) and [train_llama3.1_8b](notebooks/train_llama3.1_8b.ipynb) to fine-tune the models on the EurekaRebus dataset. The training of both models was conducted on a single RTX 3090 GPU with 24GB RAM.

### Generating and Parsing Rebus Solutions with Fine-tuned Models

To generate solutions for the rebuses in the test set, follow the instructions in the [inference](notebooks/inference.ipynb) notebook. The folder [outputs](outputs) already contains parsed [gold solutions](outputs/test_gold_id_ood.csv) and model predictions across saved training checkpoints for Phi-3 Mini and LLaMA-3.1 8B.

### Generating Rebus Solutions with Prompted LLMs

Use the [prompt_models](scritps/prompt_models.py) script to generate solutions for the rebuses in the test set using prompted LLMs. The script requires the `guidance` library to be installed. The following command generates solutions for the test set using the `Claude 3.5 Sonnet` model:

```shell
python scripts/prompt_models.py \
  --model claude \
  --api_key YOUR_ANTHROPIC_API_KEY
```

Resulting files have the same format as the ones produced by the inference notebook, and are saved in the `outputs/prompted_models` folder.

⚠️ Due to an incompatibility of the `guidance` library, we used `guidance=0.1.14` for `TogetherAI` models (LLaMA, Qwen) and `guidance=0.1.15` for proprietary models (Claude, GPT-4o). We don't guarantee that other versions will work as expected.

### Evaluating Results

For a new file containing the parsed predictions for a model produced with the inference notebook, you can compute all metrics from the paper using the following command:

```shell
python scripts/evaluate.py \
  --predicted_outputs outputs/phi3_mini/phi3_mini_results_step_500.csv \
  --gold_outputs outputs/test_gold_id_ood.csv \
  --word_frequencies outputs/word_frequencies_paisa.json \
  --word_frequencies_fp_train eureka-rebus/word_frequencies_fp_train.json \
  --word_frequencies_solution_train eureka-rebus/word_frequencies_solution_train.json \
  --do_corrs
```

To use the command above, if you use the `do_corrs` option, you should unzip the `word_frequencies_paisa.json.zip` file in the `outputs` folder.

### Acknowledgments

We would like to thank the [Associazione Culturale "Biblioteca Enigmistica Italiana - G. Panini"](http://www.enignet.it/home) for their valuable contribution in maintaining the Eureka5 collection up to date and openly accessible.
