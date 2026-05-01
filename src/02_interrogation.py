import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def setup_metal_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def interrogate_dataset(dataset_path, output_path, model, tokenizer, device):
    with open(dataset_path, "r") as f:
        data = json.load(f)
        
    print(f"Interrogating {len(data)} questions from {dataset_path}...")
    
    for item in tqdm(data):
        chat = [{"role": "user", "content": f"Answer the following trivia question briefly: {item['question']}"}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=5 
            )
        
        variations = []
        for i in range(5):
            text = tokenizer.decode(outputs[i][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            variations.append(text)
            
        item['variations'] = variations

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

def main():
    device = setup_metal_device()
    model_id = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

    interrogate_dataset("data/processed/dataset_correct.json", "data/processed/interrogated_correct.json", model, tokenizer, device)
    interrogate_dataset("data/processed/dataset_failures.json", "data/processed/interrogated_failures.json", model, tokenizer, device)
    
    print("\nPhase 2 Complete! 5 variations generated for all questions.")

if __name__ == "__main__":
    main()
