# Silent-Failures-in-small-language-models

## Overview
Small, open-weight Large Language Models (LLMs) are incredibly capable, but they suffer from a critical safety flaw: "Silent Failures." When these models don't know an answer, they often hallucinate a response with extremely high mathematical confidence (Token Probability). If you rely solely on token probability, you cannot tell the difference between a factual truth and a confident lie.

This project implements a Natural Language Inference (NLI) pipeline to calculate **Semantic Entropy**. By forcing the model to answer the same question multiple times and measuring the consistency of the *meaning* of its answers, we can successfully catch confident hallucinations before they reach the user.

## Models and Datasets
* **Target LLM:** `google/gemma-2-2b-it` (A highly capable sub-3B parameter model).
* **NLI Judge:** `cross-encoder/nli-deberta-v3-base` (Used to cluster answers by semantic meaning).
* **Dataset:** The validation split of `TriviaQA` (Subset of 500 questions).

## The Core Concept
1. **The Baseline:** Ask the model a question at Temperature = 0.0. It gives its best "first instinct" answer.
2. **The Interrogation:** Ask the model the exact same question 5 more times, but at Temperature = 1.0. 
3. **The Judge:** Use DeBERTa to read the 5 variations. If the model is telling the truth, all 5 variations will mean the same thing (Low Semantic Entropy). If the model is hallucinating, it will invent 5 completely different answers (High Semantic Entropy).

## Installation and Dependencies
This project was developed and optimized for Apple Silicon (M-Series Macs) using the `mps` backend, but it will automatically fall back to `cuda` or `cpu` depending on your hardware.

**Prerequisites:**
You will need a Hugging Face account and an access token to download the Gemma-2B model.

**Dependencies:**
`pip install torch transformers sentence-transformers datasets scikit-learn matplotlib numpy tqdm`

## Project Structure and Usage
The pipeline is broken down into three sequential scripts to protect system memory. Run them in this exact order:

### 1. Data Preparation (Failure Mining)
`python src/01_data_prep.py`
* **What it does:** Downloads the TriviaQA dataset, asks Gemma the questions at T=0.0, and grades the answers. It splits the data into two JSON files: `dataset_correct.json` (Truths) and `dataset_failures.json` (Hallucinations).

### 2. The Interrogation Phase
`python src/02_interrogation.py`
* **What it does:** Loads the sorted datasets and asks Gemma every question 5 more times at T=1.0. It saves the results into a new `processed/` directory.

### 3. Advanced Evaluation and Graphing
`python src/03_advanced_evaluation.py`
* **What it does:** This is the master evaluation script. It calculates the raw Token Probability, the Lexical Entropy (strict string matching), and the Semantic Entropy (using the DeBERTa NLI Judge). It then calculates the AUROC scores and draws the final comparative graphs.

## Expected Results
When you run the final evaluation script, you will receive terminal outputs displaying the AUROC scores, proving that Semantic Entropy significantly outperforms raw Token Probability at detecting lies.

The script will also generate and save two visual proofs in the `results/` folder:
1. **Semantic Entropy Distribution (Histogram):** A graph showing the H-Scores of the Truths (Green) versus the Hallucinations (Red). You will see two distinct clusters, visually proving the method works.
2. **Accuracy-Rejection Curve:** A line graph demonstrating that if you build a safety filter based on Semantic Entropy, the accuracy of the remaining answers climbs significantly higher and faster than if you filtered by Token Probability or Lexical Entropy.
