import torch
import json
import gc
import math
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import CrossEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def setup_metal_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def clear_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ==========================================
# MATHEMATICAL FORMULAS
# ==========================================
def calculate_shannon_entropy(probabilities):
    """
    The manual Shannon Entropy Formula: H = -Sum( P(x) * log(P(x)) )
    """
    return -sum(p * math.log(p) for p in probabilities if p > 0)

# ==========================================
# PHASE 1: RETRIEVE METRICS
# ==========================================
def process_dataset(filepath, model, tokenizer, judge, device):
    with open(filepath, "r") as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Processing {filepath.split('/')[-1]}"):
        # 1. Get Token Probability
        chat = [{"role": "user", "content": f"Answer the following trivia question briefly: {item['question']}"}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False, return_dict_in_generate=True, output_scores=True)
            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            item['token_probability'] = torch.exp(transition_scores[0].mean()).item() if transition_scores is not None else 0.5

        variations = item['variations']

        # 2. Get Lexical Entropy
        lex_clusters = {}
        for v in variations:
            clean_v = v.lower().strip()
            lex_clusters[clean_v] = lex_clusters.get(clean_v, 0) + 1
        lex_probs = [count / len(variations) for count in lex_clusters.values()]
        item['lexical_entropy'] = calculate_shannon_entropy(lex_probs)

        # 3. Get Semantic Entropy
        sem_clusters = []
        for ans in variations:
            added = False
            for cluster in sem_clusters:
                if judge.predict([cluster[0], ans]).argmax() == 1:
                    cluster.append(ans)
                    added = True
                    break
            if not added:
                sem_clusters.append([ans])
        
        sem_probs = [len(c) / len(variations) for c in sem_clusters]
        item['semantic_entropy'] = calculate_shannon_entropy(sem_probs)

    return data

# ==========================================
# PHASE 2: EVALUATION & GRAPHING
# ==========================================
def draw_accuracy_rejection_curve(uncertainties, accuracies, metric_name, color):
    sorted_indices = np.argsort(uncertainties)
    sorted_acc = np.array(accuracies)[sorted_indices]
    
    rejection_rates, acc_at_rejection = [], []
    for i in range(0, 95, 5): 
        rejection_rate = i / 100.0
        keep_count = int(len(sorted_acc) * (1 - rejection_rate))
        if keep_count > 0:
            rejection_rates.append(i)
            acc_at_rejection.append(np.mean(sorted_acc[:keep_count]))
            
    plt.plot(rejection_rates, acc_at_rejection, label=metric_name, color=color, marker='o', markersize=4)

def generate_graphs(correct_data, failures_data):
    all_data = correct_data + failures_data
    y_true = np.array([0] * len(correct_data) + [1] * len(failures_data))
    accuracies = np.array([1] * len(correct_data) + [0] * len(failures_data))
    
    tok_uncertainty = 1.0 - np.array([item['token_probability'] for item in all_data])
    lex_ent = np.array([item['lexical_entropy'] for item in all_data])
    sem_ent = np.array([item['semantic_entropy'] for item in all_data])

    print("\nFINAL METRICS: Gemma-2B")
    print(f"Token Probability AUROC: {roc_auc_score(y_true, tok_uncertainty):.4f}")
    print(f"Lexical Entropy AUROC:   {roc_auc_score(y_true, lex_ent):.4f}")
    print(f"Semantic Entropy AUROC:  {roc_auc_score(y_true, sem_ent):.4f}")

    # GRAPH 1: Original Histogram
    plt.figure(figsize=(10, 6))
    correct_sem = [item['semantic_entropy'] for item in correct_data]
    failure_sem = [item['semantic_entropy'] for item in failures_data]
    plt.hist(correct_sem, bins=15, alpha=0.6, color='green', label='Truths (Correct)', edgecolor='black')
    plt.hist(failure_sem, bins=15, alpha=0.6, color='red', label='Hallucinations (Failures)', edgecolor='black')
    plt.axvline(x=np.mean(correct_sem), color='darkgreen', linestyle='dashed', linewidth=2)
    plt.axvline(x=np.mean(failure_sem), color='darkred', linestyle='dashed', linewidth=2)
    plt.title('Semantic Entropy Distribution', fontsize=16)
    plt.xlabel('H-Score (Semantic Entropy)', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('results/original_histogram.png', dpi=300)
    plt.show()

    # GRAPH 2: Accuracy-Rejection Curve
    plt.figure(figsize=(9, 6))
    draw_accuracy_rejection_curve(tok_uncertainty, accuracies, "Token Probability", "blue")
    draw_accuracy_rejection_curve(lex_ent, accuracies, "Lexical Entropy", "orange")
    draw_accuracy_rejection_curve(sem_ent, accuracies, "Semantic Entropy", "green")
    plt.title('Accuracy-Rejection Curve: Gemma-2B', fontsize=16)
    plt.xlabel('Rejection Rate (%)', fontsize=12)
    plt.ylabel('Accuracy of Remaining Answers', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/accuracy_rejection.png', dpi=300)
    plt.show()

def main():
    device = setup_metal_device()
    
    print("Loading Gemma to calculate Token Probabilities...")
    model_id = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    
    print("Loading DeBERTa to calculate Semantic Entropy...")
    judge = CrossEncoder('cross-encoder/nli-deberta-v3-base', device=device)

    correct_data = process_dataset("data/processed/interrogated_correct.json", model, tokenizer, judge, device)
    failures_data = process_dataset("data/processed/interrogated_failures.json", model, tokenizer, judge, device)

    with open("data/processed/final_scored_correct.json", "w") as f: json.dump(correct_data, f, indent=4)
    with open("data/processed/final_scored_failures.json", "w") as f: json.dump(failures_data, f, indent=4)

    del model, tokenizer, judge
    clear_memory()

    print("\nGenerating Graphs...")
    generate_graphs(correct_data, failures_data)

if __name__ == "__main__":
    main()
