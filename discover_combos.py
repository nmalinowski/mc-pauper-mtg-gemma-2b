import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from itertools import combinations

MODEL_PATH = "./gemma-mtg-combo-finder"
BASE_MODEL = "google/gemma-2b-it"


def load_model():
    """Load the fine-tuned model"""
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, instruction, input_text, max_tokens=512):
    """Generate response with reasoning"""
    prompt = f"""<start_of_turn>user
{instruction}

{input_text}<end_of_turn>
<start_of_turn>model
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,  # Lower temp for more focused reasoning
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<start_of_turn>model")[-1].strip()

    return response


def analyze_card_pair(model, tokenizer, card1, card2):
    """Analyze if two cards have combo potential"""
    instruction = (
        "Analyze if these two cards create a combo or synergy in Pauper format."
    )
    input_text = f"""Card 1: {card1['name']}
Mana Cost: {card1['mana_cost']}
Type: {card1['type_line']}
Text: {card1['oracle_text']}

Card 2: {card2['name']}
Mana Cost: {card2['mana_cost']}
Type: {card2['type_line']}
Text: {card2['oracle_text']}"""

    return generate_response(model, tokenizer, instruction, input_text)


def check_for_infinite(model, tokenizer, cards):
    """Check if a set of cards creates an infinite combo"""
    instruction = "Analyze if these cards create an infinite combo in Pauper format. Think step-by-step."

    card_descriptions = []
    for card in cards:
        card_descriptions.append(f"{card['name']}: {card['oracle_text']}")

    input_text = "Cards:\n" + "\n\n".join(card_descriptions)

    return generate_response(model, tokenizer, instruction, input_text, max_tokens=768)


def discover_new_combos(model, tokenizer, cards, known_combos):
    """Search for novel combos by testing card combinations"""
    print("\n" + "=" * 60)
    print("SEARCHING FOR NOVEL COMBOS")
    print("=" * 60 + "\n")

    # Filter to combo-relevant cards
    combo_cards = []
    for card in cards:
        ability_count = sum(card["abilities"].values())
        if ability_count >= 2:  # Cards with multiple relevant abilities
            combo_cards.append(card)

    print(f"Analyzing {len(combo_cards)} high-potential cards...")

    known_card_sets = set()
    for combo in known_combos:
        known_card_sets.add(frozenset(combo["cards"]))

    discoveries = []

    # Test 2-card combos
    print("\nTesting 2-card combinations...")
    for i, (card1, card2) in enumerate(combinations(combo_cards[:50], 2)):
        card_set = frozenset([card1["name"], card2["name"]])

        if card_set not in known_card_sets:
            print(f"[{i+1}] Testing: {card1['name']} + {card2['name']}")

            analysis = analyze_card_pair(model, tokenizer, card1, card2)

            # Check if model identifies it as a combo
            if any(
                keyword in analysis.lower()
                for keyword in ["combo", "infinite", "synergy", "loop", "repeatedly"]
            ):
                print(f"  ✓ POTENTIAL COMBO FOUND!")
                discoveries.append(
                    {
                        "cards": [card1["name"], card2["name"]],
                        "analysis": analysis,
                        "novelty": "potentially_new",
                    }
                )
                print(f"  Analysis: {analysis[:200]}...")

    # Test 3-card combos (limited for performance)
    print("\nTesting 3-card combinations...")
    for i, (card1, card2, card3) in enumerate(combinations(combo_cards[:30], 3)):
        if i >= 20:  # Limit iterations
            break

        card_set = frozenset([card1["name"], card2["name"], card3["name"]])

        if card_set not in known_card_sets:
            print(
                f"[{i+1}] Testing: {card1['name']} + {card2['name']} + {card3['name']}"
            )

            analysis = check_for_infinite(model, tokenizer, [card1, card2, card3])

            if "infinite" in analysis.lower() or "yes" in analysis.lower()[:100]:
                print(f"  ✓ POTENTIAL INFINITE COMBO!")
                discoveries.append(
                    {
                        "cards": [card1["name"], card2["name"], card3["name"]],
                        "analysis": analysis,
                        "novelty": "potentially_new",
                    }
                )

            print(f"  Analysis: {analysis[:200]}...")
    return discoveries


def main():
    model, tokenizer = load_model()

    # Load card database
    with open("data/pauper_cards_detailed.json", "r") as f:
        cards = json.load(f)

    # Load known combos
    with open("data/known_combos.json", "r") as f:
        known_combos = json.load(f)

    print(f"Loaded {len(cards)} Pauper cards")
    print(f"Loaded {len(known_combos)} known combos")

    # Test on known combos first (validation)
    print("\n" + "=" * 60)
    print("VALIDATING ON KNOWN COMBOS")
    print("=" * 60 + "\n")

    for combo in known_combos[:3]:
        print(f"\nTesting: {' + '.join(combo['cards'])}")
        print(f"Expected: {combo['description']}")

    card_objs = [c for c in cards if c["name"] in combo["cards"]]
    if len(card_objs) >= 2:
        analysis = check_for_infinite(model, tokenizer, card_objs)
        print(f"\nModel Analysis:\n{analysis}\n")
        print("-" * 60)

    # Discover new combos
    discoveries = discover_new_combos(model, tokenizer, cards, known_combos)

    # Save discoveries
    with open("data/discovered_combos.json", "w") as f:
        json.dump(discoveries, f, indent=2)
    print(f"\n{'='*60}")
    print(f"DISCOVERY COMPLETE")
    print(f"{'='*60}")
    print(f"\nTotal discoveries: {len(discoveries)}")
    print(f"Results saved to: data/discovered_combos.json")
    # Show top discoveries
    print("\n" + "=" * 60)
    print("TOP DISCOVERIES")
    print("=" * 60 + "\n")
    for i, discovery in enumerate(discoveries[:5], 1):
        print(f"{i}. {' + '.join(discovery['cards'])}")
        print(f"   {discovery['analysis'][:300]}...\n")


if __name__ == "__main__":
    main()
