import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = "./gemma-mtg-pauper"
BASE_MODEL = "google/gemma-2b-it"


def load_model():
    """Load the fine-tuned model"""
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, instruction, input_text):
    """Generate a response from the model"""
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    response = response.split("### Response:")[-1].strip()

    return response


def main():
    model, tokenizer = load_model()

    # Test queries
    test_queries = [
        {
            "instruction": "Suggest cards for a Pauper deck archetype.",
            "input": "I want to build a blue control deck in Pauper. What cards should I include?",
        },
        {
            "instruction": "Identify potential combos in Pauper.",
            "input": "What are some infinite combos possible in Pauper format?",
        },
        {
            "instruction": "Analyze this Magic: The Gathering Pauper format card.",
            "input": "What makes Counterspell a good card in Pauper?",
        },
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query['input']}")
        print(f"{'='*60}")
        response = generate_response(
            model, tokenizer, query["instruction"], query["input"]
        )
        print(f"RESPONSE: {response}\n")


if __name__ == "__main__":
    main()
