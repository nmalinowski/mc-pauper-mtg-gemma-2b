import json
import requests
import pandas as pd
from pathlib import Path

def fetch_pauper_cards():
    """Fetch all Pauper-legal cards from Scryfall API"""
    print("Fetching Pauper-legal cards from Scryfall...")
    
    all_cards = []
    url = "https://api.scryfall.com/cards/search"
    params = {
        "q": "legal:pauper",
        "format": "json",
        "unique": "prints"
    }
    
    while url:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
            
        data = response.json()
        all_cards.extend(data.get("data", []))
        url = data.get("next_page")
        params = None  # Only use params on first request
        print(f"Fetched {len(all_cards)} cards so far...")
    
    return all_cards

def process_card_data(cards):
    """Extract relevant information for training"""
    processed_cards = []
    
    for card in cards:
        card_info = {
            "name": card.get("name"),
            "mana_cost": card.get("mana_cost", ""),
            "cmc": card.get("cmc", 0),
            "type_line": card.get("type_line"),
            "oracle_text": card.get("oracle_text", ""),
            "colors": card.get("colors", []),
            "color_identity": card.get("color_identity", []),
            "power": card.get("power"),
            "toughness": card.get("toughness"),
            "keywords": card.get("keywords", []),
            "rarity": card.get("rarity"),
        }
        processed_cards.append(card_info)
    
    return processed_cards

def create_training_examples(cards):
    """Create training examples for deck building and combo identification"""
    examples = []
    
    # Example 1: Card description and analysis
    for card in cards:
        if card["oracle_text"]:
            prompt = f"Analyze this Pauper card:\nName: {card['name']}\nMana Cost: {card['mana_cost']}\nType: {card['type_line']}\nText: {card['oracle_text']}"
            response = f"This is a {card['type_line']} that costs {card['mana_cost']}. "
            
            # Add context based on card type
            if "Creature" in card["type_line"]:
                response += f"It's a creature with {card['power']}/{card['toughness']} stats. "
            
            if card["keywords"]:
                response += f"It has the following abilities: {', '.join(card['keywords'])}. "
            
            examples.append({
                "instruction": "Analyze this Magic: The Gathering Pauper format card.",
                "input": prompt,
                "output": response
            })
    
    # Example 2: Deck building queries
    color_combos = [
        (["U"], "blue control"),
        (["R"], "red aggro"),
        (["B"], "black sacrifice"),
        (["W"], "white weenie"),
        (["G"], "green ramp"),
        (["U", "B"], "Dimir control"),
        (["R", "G"], "Gruul aggro"),
    ]
    
    for colors, archetype in color_combos:
        matching_cards = [c for c in cards if set(colors).issubset(set(c.get("color_identity", [])))]
        if len(matching_cards) > 10:
            card_list = [c["name"] for c in matching_cards[:20]]
            examples.append({
                "instruction": "Suggest cards for a Pauper deck archetype.",
                "input": f"I want to build a {archetype} deck in Pauper. What cards should I consider?",
                "output": f"For a {archetype} deck, consider these Pauper-legal cards: {', '.join(card_list[:15])}. These cards synergize well with the {archetype} strategy."
            })
    
    return examples

# Main execution
if __name__ == "__main__":
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Fetch cards
    cards = fetch_pauper_cards()
    print(f"\nTotal cards fetched: {len(cards)}")
    
    # Process cards
    processed = process_card_data(cards)
    
    # Save card database
    with open("data/pauper_cards.json", "w") as f:
        json.dump(processed, f, indent=2)
    
    # Create training examples
    training_examples = create_training_examples(processed)
    print(f"Created {len(training_examples)} training examples")
    
    # Save training data
    with open("data/training_data.json", "w") as f:
        json.dump(training_examples, f, indent=2)
    
    print("\nData collection complete!")
    print(f"- Pauper cards saved to: data/pauper_cards.json")
    print(f"- Training data saved to: data/training_data.json")