import argparse
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Ensure NLTK data is downloaded
for res in ["wordnet", "omw-1.4", "averaged_perceptron_tagger", "punkt"]:
    try: nltk.download(res, quiet=True)
    except Exception: pass

MODELS = {
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B"
}

def main():
    parser = argparse.ArgumentParser(description="Run Input Transformations LTO Pipeline")
    parser.add_argument("--model", type=str, choices=MODELS.keys(), default="llama3.2-1b", help="Model to evaluate")
    parser.add_argument("--dataset", type=str, default="commonsense_qa", help="Dataset to test")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[args.model], 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    
    print("Model loaded successfully. Ready to run evaluations.")
    # Import your datasets and execute run_multi() logic here using the loaded model...

if __name__ == "__main__":
    main()
