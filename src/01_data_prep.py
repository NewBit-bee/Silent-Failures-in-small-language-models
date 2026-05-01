import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import os
from tqdm import tqdm

def setup_metal_device():
    """Maps PyTorch to the Mac M4 GPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def check_answer(generation, aliases):
    """Auto-grader: Checks if any verified alias is inside the generated text."""
    gen_lower = generation.lower()
    for alias in aliases:
        if alias.lower() in gen_lower:
            return True
    return False

def main():
    device = setup_metal_device()
    print(f"Initializing on device: {device}")

    # 1. Load Model (Unquantized, Native 16-bit for full research accuracy)
    model_id = "google/gemma-2-2b-it"
    print(f"Loading {model_id} into M4 Memory...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # 2. Load TriviaQA Dataset
    print("Loading TriviaQA dataset...")
    dataset = load_dataset("trivia_qa", "rc", split="validation")
    
    # Using a subset of 500 questions for the initial baseline mining
    subset = dataset.select(range(500))

    dataset_correct = []
    dataset_failures = []

    print("Starting Phase 1: Failure Mining (Temp=0.0)")
    for item in tqdm(subset):
        question = item['question']
        aliases = item['answer']['normalized_aliases']

        # Format prompt for Gemma Instruct
        chat = [{"role": "user", "content": f"Answer the following trivia question briefly: {question}"}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate deterministically
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.0, # Crucial for Phase 1 baseline
                do_sample=False
            )
        
        # Decode the output (excluding the prompt)
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        # 3. Auto-Grade
        is_correct = check_answer(generated_text, aliases)

        record = {
            "question_id": item['question_id'],
            "question": question,
            "generated_answer": generated_text,
            "ground_truth_aliases": aliases
        }

        if is_correct:
            dataset_correct.append(record)
        else:
            dataset_failures.append(record)

    # 4. Save Outputs
    os.makedirs("data/processed", exist_ok=True)
    
    with open("data/processed/dataset_correct.json", "w") as f:
        json.dump(dataset_correct, f, indent=4)
        
    with open("data/processed/dataset_failures.json", "w") as f:
        json.dump(dataset_failures, f, indent=4)

    print("\nPhase 1 Complete!")
    print(f"Factual Truths Found: {len(dataset_correct)}")
    print(f"Confident Lies (Failures) Found: {len(dataset_failures)}")

if __name__ == "__main__":
    main()
