import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

MODEL_PATH = "./gemma-mtg-combo-finder"
BASE_MODEL = "google/gemma-2b-it"


class ComboExplorer:
    def __init__(self):
        print("Loading Pauper Combo Finder...")
        self.model, self.tokenizer = self.load_model()
        self.cards = self.load_cards()
        print(f"✓ Ready! Loaded {len(self.cards)} Pauper cards\n")

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model.eval()
        return model, tokenizer

    def load_cards(self):
        with open("data/pauper_cards_detailed.json", "r") as f:
            return json.load(f)

    def generate(self, instruction, input_text):
        prompt = f"""<start_of_turn>user
{instruction}

{input_text}<end_of_turn>
<start_of_turn>model
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=512, temperature=0.3, top_p=0.9, do_sample=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("<start_of_turn>model")[-1].strip()

    def find_card(self, name):
        for card in self.cards:
            if card["name"].lower() == name.lower():
                return card
        return None

    def analyze_combo(self, card_names):
        cards_found = []
        for name in card_names:
            card = self.find_card(name)
            if card:
                cards_found.append(card)
            else:
                print(f"Warning: Card '{name}' not found in Pauper format")

        if len(cards_found) < 2:
            return "Need at least 2 valid Pauper cards to analyze"

        card_descriptions = []
        for card in cards_found:
            card_descriptions.append(
                f"{card['name']} ({card['mana_cost']})\n"
                f"Type: {card['type_line']}\n"
                f"Text: {card['oracle_text']}"
            )

        input_text = "Cards:\n\n" + "\n\n".join(card_descriptions)
        instruction = "Analyze if these cards create an infinite combo or powerful synergy in Pauper format. Explain step-by-step."

        return self.generate(instruction, input_text)

    def suggest_combo_pieces(self, card_name):
        card = self.find_card(card_name)
        if not card:
            return f"Card '{card_name}' not found in Pauper format"

        instruction = "What cards would combo well with this card in Pauper format?"
        input_text = f"""{card['name']} ({card['mana_cost']})
Type: {card['type_line']}
Text: {card['oracle_text']}"""

        return self.generate(instruction, input_text)

    def interactive_mode(self):
        print("=" * 60)
        print("PAUPER COMBO EXPLORER - INTERACTIVE MODE")
        print("=" * 60)
        print("\nCommands:")
        print("  combo <card1>, <card2>, [card3]  - Analyze combo potential")
        print("  suggest <card>                    - Get combo suggestions")
        print("  quit                              - Exit")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("\n> ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if user_input.lower().startswith("combo "):
                    card_names = [c.strip() for c in user_input[6:].split(",")]
                    print(f"\nAnalyzing: {' + '.join(card_names)}\n")
                    result = self.analyze_combo(card_names)
                    print(result)

                elif user_input.lower().startswith("suggest "):
                    card_name = user_input[8:].strip()
                    print(f"\nFinding combos for: {card_name}\n")
                    result = self.suggest_combo_pieces(card_name)
                    print(result)

                else:
                    print("Unknown command. Use 'combo <cards>' or 'suggest <card>'")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    explorer = ComboExplorer()

    # Quick test
    print("Testing known combo...")
    result = explorer.analyze_combo(["Midnight Guard", "Presence of Gond"])
    print(result)

    print("\n" + "=" * 60 + "\n")

    # Interactive mode
    explorer.interactive_mode()


if __name__ == "__main__":
    main()
