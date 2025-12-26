import json
import requests
import re
from pathlib import Path
from itertools import combinations
import time


def fetch_pauper_cards():
    """Fetch all Pauper-legal cards with detailed information"""
    print("Fetching Pauper-legal cards...")

    all_cards = []
    url = "https://api.scryfall.com/cards/search"
    params = {
        "q": "legal:pauper",
        "format": "json",
        "unique": "cards",  # Get unique cards, not all prints
    }

    while url:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        data = response.json()
        all_cards.extend(data.get("data", []))
        url = data.get("next_page")
        params = None
        print(f"Fetched {len(all_cards)} cards...")
        time.sleep(0.1)  # Rate limiting

    return all_cards


def extract_card_features(card):
    """Extract detailed features for combo detection"""
    oracle_text = card.get("oracle_text", "")

    features = {
        "name": card.get("name"),
        "mana_cost": card.get("mana_cost", ""),
        "cmc": card.get("cmc", 0),
        "type_line": card.get("type_line", ""),
        "oracle_text": oracle_text,
        "colors": card.get("colors", []),
        "color_identity": card.get("color_identity", []),
        "keywords": card.get("keywords", []),
        "power": card.get("power"),
        "toughness": card.get("toughness"),
        # Combo-relevant features
        "abilities": {
            "enters_battlefield": "enters the battlefield" in oracle_text.lower()
            or "enter the battlefield" in oracle_text.lower(),
            "leaves_battlefield": "leaves the battlefield" in oracle_text.lower()
            or "leave the battlefield" in oracle_text.lower(),
            "dies": "dies" in oracle_text.lower()
            or "when {name} dies".format(name=card.get("name", ""))
            in oracle_text.lower(),
            "draw": "draw" in oracle_text.lower(),
            "untap": "untap" in oracle_text.lower(),
            "tap_ability": "{T}:" in oracle_text or "tap:" in oracle_text.lower(),
            "sacrifice": "sacrifice" in oracle_text.lower(),
            "return_to_hand": "return" in oracle_text.lower()
            and "hand" in oracle_text.lower(),
            "flicker": "exile" in oracle_text.lower()
            and "return" in oracle_text.lower(),
            "create_token": "create" in oracle_text.lower()
            and "token" in oracle_text.lower(),
            "add_mana": "add {" in oracle_text.lower()
            or "add one mana" in oracle_text.lower(),
            "storm": "storm" in oracle_text.lower(),
            "cost_reduction": "cost" in oracle_text.lower()
            and ("less" in oracle_text.lower() or "reduced" in oracle_text.lower()),
            "bounce": "return" in oracle_text.lower()
            and (
                "owner's hand" in oracle_text.lower()
                or "its owner's hand" in oracle_text.lower()
            ),
            "copy_spell": "copy" in oracle_text.lower()
            and (
                "spell" in oracle_text.lower()
                or "instant" in oracle_text.lower()
                or "sorcery" in oracle_text.lower()
            ),
            "tutor": "search your library" in oracle_text.lower(),
            "recur": (
                "from your graveyard" in oracle_text.lower()
                and "to" in oracle_text.lower()
            )
            or "return" in oracle_text.lower()
            and "graveyard" in oracle_text.lower(),
        },
        # Card type flags
        "is_creature": "Creature" in card.get("type_line", ""),
        "is_instant": "Instant" in card.get("type_line", ""),
        "is_sorcery": "Sorcery" in card.get("type_line", ""),
        "is_artifact": "Artifact" in card.get("type_line", ""),
        "is_enchantment": "Enchantment" in card.get("type_line", ""),
        "is_land": "Land" in card.get("type_line", ""),
    }

    return features


def scrape_known_combos():
    """
    Scrape known Pauper combos from various sources
    You'll need to manually curate this or scrape from MTG sites
    """
    # This is a starter set - you should expand this significantly
    known_combos = [
        {
            "cards": ["Ghostly Flicker", "Archaeomancer", "Mnemonic Wall"],
            "description": "Infinite mana with land untap",
            "steps": [
                "Have Archaeomancer or Mnemonic Wall on the battlefield",
                "Cast Ghostly Flicker targeting the creature and a land",
                "Flicker returns Ghostly Flicker to your hand",
                "Untap the land to generate mana",
                "Repeat for infinite mana and flickers",
            ],
            "requirements": [
                "2 creatures that return instants/sorceries",
                "Ghostly Flicker",
                "lands for mana",
            ],
            "result": "Infinite mana, infinite ETB triggers, infinite flicker effects",
            "type": "infinite",
        },
        {
            "cards": ["Famished Paladin", "Presence of Gond", "Soul Warden"],
            "description": "Infinite creature tokens and life",
            "steps": [
                "Enchant Famished Paladin with Presence of Gond",
                "Have Soul Warden on battlefield",
                "Tap Famished Paladin to create an Elf token with Presence of Gond",
                "Soul Warden triggers, you gain 1 life",
                "Famished Paladin untaps from life gain",
                "Repeat for infinite tokens and life",
            ],
            "requirements": [
                "Famished Paladin",
                "Presence of Gond",
                "Soul Warden or similar life gain trigger",
            ],
            "result": "Infinite 1/1 Elf tokens, infinite life gain",
            "type": "infinite",
        },
        {
            "cards": ["Midnight Guard", "Presence of Gond"],
            "description": "Infinite creature tokens",
            "steps": [
                "Enchant Midnight Guard with Presence of Gond",
                "Tap Midnight Guard to create a 1/1 Elf token",
                "Midnight Guard untaps when the token enters",
                "Repeat for infinite tokens",
            ],
            "requirements": ["Midnight Guard", "Presence of Gond"],
            "result": "Infinite 1/1 Elf tokens",
            "type": "infinite",
        },
        {
            "cards": [
                "Sage's Row Denizen",
                "Mortuary Mire",
                "Mnemonic Wall",
                "Ghostly Flicker",
            ],
            "description": "Infinite mill combo",
            "steps": [
                "Have Sage's Row Denizen and Mnemonic Wall on battlefield",
                "Cast Ghostly Flicker targeting Mnemonic Wall and Mortuary Mire",
                "Mortuary Mire returns Ghostly Flicker to top of library",
                "Mnemonic Wall returns Ghostly Flicker to hand",
                "Sage's Row Denizen mills opponent for each ETB",
                "Repeat until opponent is milled out",
            ],
            "requirements": [
                "Sage's Row Denizen",
                "Mnemonic Wall",
                "Ghostly Flicker",
                "Mortuary Mire",
                "enough mana",
            ],
            "result": "Infinite mill, infinite ETB triggers",
            "type": "infinite",
        },
        {
            "cards": [
                "Freed from the Real",
                "Zealous Conscripts",
                "land that produces 2+ mana",
            ],
            "description": "Infinite mana combo",
            "steps": [
                "Enchant a creature that taps for mana with Freed from the Real",
                "Tap creature for 2+ mana",
                "Pay 1 blue to untap with Freed from the Real",
                "Net 1+ mana per cycle",
                "Repeat for infinite mana",
            ],
            "requirements": ["Creature that taps for 2+ mana", "Freed from the Real"],
            "result": "Infinite mana",
            "type": "infinite",
        },
        {
            "cards": ["Peregrine Drake", "Ghostly Flicker", "Archaeomancer"],
            "description": "Classic infinite mana combo",
            "steps": [
                "Have Archaeomancer and Peregrine Drake on battlefield",
                "Cast Ghostly Flicker targeting both creatures",
                "Archaeomancer returns Ghostly Flicker",
                "Peregrine Drake untaps 5 lands",
                "Net 2 mana per loop (5 untapped - 3 to cast)",
                "Repeat for infinite mana",
            ],
            "requirements": [
                "Peregrine Drake",
                "Archaeomancer or Mnemonic Wall",
                "Ghostly Flicker",
                "5 lands",
            ],
            "result": "Infinite mana, infinite ETB triggers",
            "type": "infinite",
        },
        {
            "cards": [
                "Frilled Deathspitter",
                "Guilty Conscience",
                "Loran's Escape",
                "Gut Shot",
            ],
            "description": "Infinite damage combo using indestructible and damage redirection",
            "steps": [
                "Enchant Frilled Deathspitter with Guilty Conscience",
                "Give Frilled Deathspitter indestructible using Loran's Escape, Blacksmith's Skill, or Tyr's Blessing",
                "Deal damage to Frilled Deathspitter (via Gut Shot, Lightning Bolt, combat, or blocking)",
                "Frilled Deathspitter's ability triggers, dealing 1 damage to target opponent",
                "Guilty Conscience triggers, dealing 1 damage to Frilled Deathspitter",
                "Step 4 repeats infinitely because Frilled Deathspitter has indestructible",
                "Loop continues until opponent loses all life",
            ],
            "requirements": [
                "Frilled Deathspitter on battlefield",
                "Guilty Conscience enchanting Frilled Deathspitter",
                "Indestructible protection (Loran's Escape, Blacksmith's Skill, or Tyr's Blessing)",
                "Damage source (Gut Shot, Lightning Bolt, combat damage, or blocking)",
            ],
            "result": "Infinite damage to target opponent",
            "type": "infinite",
            "mana_cost_minimum": "3-4 mana (depending on protection spell and damage source)",
            "colors": ["R", "W"],
            "notes": "Protection spell must be instant-speed to respond to damage trigger. Frilled Deathspitter's ability targets, so opponent can be changed mid-combo if needed. The combo is resilient because the protection spells also provide hexproof/protection from colors temporarily.",
        },
    ]

    return known_combos


def generate_potential_combos(cards):
    """
    Programmatically generate potential combo interactions
    This creates training data for the model to learn patterns
    """
    potential_combos = []

    # Find cards that work together based on abilities
    etb_creatures = [
        c for c in cards if c["abilities"]["enters_battlefield"] and c["is_creature"]
    ]
    flicker_effects = [c for c in cards if c["abilities"]["flicker"]]
    untap_effects = [c for c in cards if c["abilities"]["untap"]]
    token_creators = [c for c in cards if c["abilities"]["create_token"]]

    # Generate ETB + Flicker combos
    for flicker in flicker_effects[:5]:  # Limit for performance
        for etb in etb_creatures[:20]:
            if flicker["name"] != etb["name"]:
                potential_combos.append(
                    {
                        "cards": [flicker["name"], etb["name"]],
                        "synergy_type": "ETB + Flicker",
                        "description": f"{flicker['name']} can repeatedly trigger {etb['name']}'s enters-the-battlefield ability",
                        "card1_role": "Flicker effect",
                        "card2_role": "ETB trigger source",
                        "analysis": f"When you flicker {etb['name']} with {flicker['name']}, you get repeated value from: {etb['oracle_text'][:100]}...",
                    }
                )

    # Find untap + tap ability combos
    tap_creatures = [
        c for c in cards if c["abilities"]["tap_ability"] and c["is_creature"]
    ]
    for untapper in untap_effects[:5]:
        for tapper in tap_creatures[:20]:
            if untapper["name"] != tapper["name"]:
                potential_combos.append(
                    {
                        "cards": [untapper["name"], tapper["name"]],
                        "synergy_type": "Untap + Tap Ability",
                        "description": f"{untapper['name']} can untap {tapper['name']} to use its tap ability multiple times",
                        "card1_role": "Untap source",
                        "card2_role": "Tap ability",
                        "analysis": f"By untapping {tapper['name']}, you can use its ability more than once per turn",
                    }
                )

    # Find token + sacrifice synergies
    sacrifice_outlets = [c for c in cards if c["abilities"]["sacrifice"]]
    for token_maker in token_creators[:10]:
        for sac_outlet in sacrifice_outlets[:10]:
            if token_maker["name"] != sac_outlet["name"]:
                potential_combos.append(
                    {
                        "cards": [token_maker["name"], sac_outlet["name"]],
                        "synergy_type": "Token Generation + Sacrifice",
                        "description": f"{token_maker['name']} creates tokens for {sac_outlet['name']} to sacrifice",
                        "card1_role": "Token generator",
                        "card2_role": "Sacrifice outlet",
                        "analysis": "This creates value by generating tokens to fuel sacrifice effects",
                    }
                )

    return potential_combos


def create_reasoning_examples(cards, combos):
    """
    Create training examples that teach the model to REASON about combos
    This is crucial for discovering new ones
    """
    examples = []

    # Example 1: Teach combo pattern recognition
    for combo in combos:
        if combo.get("type") == "infinite":
            examples.append(
                {
                    "instruction": "Analyze if these cards create an infinite combo in Pauper format.",
                    "input": f"Cards: {', '.join(combo['cards'])}",
                    "output": f"Yes, this is an infinite combo. Here's how it works:\n\n"
                    + f"Description: {combo['description']}\n\n"
                    + f"Steps:\n"
                    + "\n".join(
                        [f"{i+1}. {step}" for i, step in enumerate(combo["steps"])]
                    )
                    + f"\n\nResult: {combo['result']}\n\n"
                    + f"Requirements: {', '.join(combo['requirements'])}",
                }
            )

    # Example 2: Teach card interaction analysis
    card_dict = {c["name"]: c for c in cards}
    for combo in combos[:10]:
        combo_cards = [
            card_dict.get(name) for name in combo["cards"] if name in card_dict
        ]
        if len(combo_cards) >= 2:
            card1, card2 = combo_cards[0], combo_cards[1]
            examples.append(
                {
                    "instruction": "Explain how these two cards interact in a Pauper combo.",
                    "input": f"Card 1: {card1['name']}\n{card1['oracle_text']}\n\nCard 2: {card2['name']}\n{card2['oracle_text']}",
                    "output": f"These cards create a synergistic interaction:\n\n{combo.get('description', 'These cards work together.')}\n\n"
                    + f"Key interaction: {combo['steps'][0] if combo.get('steps') else 'They enable each other.'}",
                }
            )

    # Example 3: Teach feature-based combo detection
    examples.append(
        {
            "instruction": "What types of cards would combo well with this card?",
            "input": "Card: Midnight Guard\nText: Whenever another creature enters the battlefield under your control, untap Midnight Guard.",
            "output": "Midnight Guard combos well with:\n\n1. Cards that create tokens when it taps (like Presence of Gond) - creates infinite tokens\n2. Cards with beneficial tap abilities\n3. Cards that generate multiple creatures\n\nThe key is that it untaps whenever a creature enters, so you want effects that trigger when you tap it AND create creatures.",
        }
    )

    examples.append(
        {
            "instruction": "Identify what makes this a combo piece.",
            "input": "Card: Ghostly Flicker\nText: Exile two target artifacts, creatures, and/or lands you control, then return those cards to the battlefield under your control.",
            "output": "Ghostly Flicker is a combo enabler because:\n\n1. It can target creatures with ETB abilities for repeated value\n2. It can target itself by flickering creatures that return instants/sorceries (like Archaeomancer)\n3. It can untap lands for mana generation\n4. It's an instant, so it has flexibility\n\nLook for creatures with 'when enters the battlefield' that return instant cards, or creatures that untap lands.",
        }
    )

    # Example 4: Teach step-by-step reasoning
    examples.append(
        {
            "instruction": "Walk through how to identify if cards form an infinite loop.",
            "input": "I have Famished Paladin (untaps when I gain life), Presence of Gond (enchantment that lets me tap to create a 1/1 elf), and Soul Warden (gain 1 life when creature enters). Is this infinite?",
            "output": "Let me analyze this step-by-step:\n\n"
            + "Step 1: Identify the trigger - Famished Paladin untaps when you gain life\n"
            + "Step 2: Identify the action - Presence of Gond lets Paladin tap to create a token\n"
            + "Step 3: Identify the loop closer - Soul Warden gains life when creature enters\n"
            + "Step 4: Test the loop:\n"
            + "  - Tap Paladin → Create Elf token\n"
            + "  - Elf enters → Soul Warden triggers, gain 1 life\n"
            + "  - Gain life → Famished Paladin untaps\n"
            + "  - Loop back to step 1\n\n"
            + "Conclusion: YES, this is an infinite combo creating infinite 1/1 elves and infinite life.",
        }
    )

    return examples


def main():
    Path("data").mkdir(exist_ok=True)

    # Fetch all Pauper cards
    cards = fetch_pauper_cards()
    print(f"\nTotal Pauper cards: {len(cards)}")

    # Extract features
    card_features = [extract_card_features(c) for c in cards]

    # Save card database
    with open("data/pauper_cards_detailed.json", "w") as f:
        json.dump(card_features, f, indent=2)

    # Get known combos
    known_combos = scrape_known_combos()
    print(f"Known combos: {len(known_combos)}")

    # Generate potential synergies
    print("Generating potential combos...")
    potential_combos = generate_potential_combos(card_features)
    print(f"Potential combos generated: {len(potential_combos)}")

    # Create reasoning examples
    print("Creating training examples...")
    reasoning_examples = create_reasoning_examples(card_features, known_combos)

    # Create card analysis examples
    card_examples = []
    for card in card_features[:100]:  # Sample
        card_examples.append(
            {
                "instruction": "Analyze this Pauper card for combo potential.",
                "input": f"Card: {card['name']}\nMana Cost: {card['mana_cost']}\nType: {card['type_line']}\nText: {card['oracle_text']}",
                "output": f"Card Analysis:\n\nType: {card['type_line']}\n"
                + f"Key Abilities: {', '.join([k for k, v in card['abilities'].items() if v])}\n\n"
                + f"Combo Potential: "
                + (
                    "High"
                    if sum(card["abilities"].values()) > 2
                    else "Medium" if sum(card["abilities"].values()) > 0 else "Low"
                ),
            }
        )

    # Combine all training data
    all_training_data = reasoning_examples + card_examples

    # Save
    with open("data/combo_training_data.json", "w") as f:
        json.dump(all_training_data, f, indent=2)

    with open("data/known_combos.json", "w") as f:
        json.dump(known_combos, f, indent=2)

    with open("data/potential_combos.json", "w") as f:
        json.dump(potential_combos, f, indent=2)

    print(f"\n✓ Data collection complete!")
    print(f"  - Cards: {len(card_features)}")
    print(f"  - Known combos: {len(known_combos)}")
    print(f"  - Potential combos: {len(potential_combos)}")
    print(f"  - Training examples: {len(all_training_data)}")


if __name__ == "__main__":
    main()
