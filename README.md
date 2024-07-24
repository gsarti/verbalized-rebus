# Non Verbis, Sed Rebus: Large Language Models are Weak Solvers of Italian Rebuses

[Gabriele Sarti](https://gsarti.com) • [Tommaso Caselli](https://scholar.google.com/citations?user=fxQvP_QAAAAJ) • [Malvina Nissim](https://malvinanissim.github.io/) • [Arianna Bisazza](https://www.cs.rug.nl/~bisazza/)

<p float="left">
    <img src="img/verbalized_rebus.png" alt="An example of a verbalized rebus" width="300"/>
    <img src="img/llm_generations.png" alt="Example generations from LLMs" width="300"/>
</p>
> **Abstract:** Rebuses are puzzles requiring constrained multi-step reasoning to identify a hidden phrase from a set of images and letters. In this work, we introduce a large collection of verbalized rebuses for the Italian language and use it to assess the rebus-solving capabilities of state-of-the-art large language models. While general-purpose systems such as LLaMA-3 and GPT-4o perform poorly on this task, ad-hoc fine-tuning seems to improve models' performance. However, we find that performance gains from training are largely motivated by memorization. Our results suggest that rebus solving remains a challenging test bed to evaluate large language models' linguistic proficiency and sequential instruction-following skills.

This repository contains scripts and notebooks associated to the paper ["Non Verbis, Sed Rebus: Large Language Models are Weak Solvers of Italian Rebuses"](TBD). If you use any of the following contents for your work, we kindly ask you to cite our paper:

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

To run training and inference scripts you will need a machine with access to a GPU (required by [Unsloth](https://github.com/unslothai/unsloth)). To setup the environment for training, you can use the following command after installing the requirements, with the virtual environment activated:

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

## Reproducing the results

### Data preprocessing

TBD

### Fine-tuning LLMs on EurekaRebus

TBD

### Evaluating Results

TBD
