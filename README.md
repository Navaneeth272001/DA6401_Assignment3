# DA6401_Assignment3
The repository contains solutions to the Assignment 3.

# Sequence-to-Sequence Model with Attention in PyTorch

This project implements a character-level Sequence-to-Sequence (Seq2Seq) model with attention in PyTorch. The model is trained using a configurable training loop integrated with [Weights & Biases](https://wandb.ai/) for hyperparameter tracking and model evaluation. Additionally, it includes tools to visualize attention weights, helping to understand which parts of the input the decoder attends to when generating each output character.

## Features

- Customizable encoder-decoder architecture using LSTM or GRU cells
- Attention mechanism for improved sequence alignment and interpretability
- Configurable training via Weights & Biases sweeps
- Real-time logging of loss, accuracy, and model checkpoints
- Attention heatmap visualization for interpretability

## Requirements

- Python >= 3.7
- PyTorch >= 1.10
- wandb
- matplotlib
- seaborn
- tqdm

Install dependencies via:

```bash
pip install -r requirements.txt


