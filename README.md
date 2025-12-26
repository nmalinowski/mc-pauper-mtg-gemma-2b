# Gemma MTG Pauper Combo Finder & Deck Architect

This project utilizes **Google's Gemma (2B-it)** Large Language Model, fine-tuned on Magic: The Gathering data, to act as a specialized assistant for the **Pauper** format.

Unlike standard card search engines, this model is trained to **reason** about card mechanics. It breaks down card interaction text to identify infinite loops, synergystic engines, and deck-building strategies.

## 🚀 Project Overview

The pipeline consists of four distinct stages:

1. **Data Extraction:** Scrapes Scryfall for all Pauper-legal cards and extracts "combo-relevant" features (ETB triggers, untap effects, sacrifice outlets).
2. **Dataset Construction:** Generates synthetic "reasoning" examples that teach the model *how* to identify a combo (e.g., "If A untaps B, and B produces mana...").
3. **Fine-Tuning (LoRA):** Trains a localized adapter for the Gemma model using 4-bit quantization for efficiency.
4. **Discovery & Inference:** a CLI for chatting with the model and an automated script to brute-force check card pairs for undiscovered combos.

---

## 🛠 Prerequisites

### Hardware

* **GPU:** NVIDIA GPU with at least **6GB VRAM** (for Gemma-2B) or **16GB VRAM** (for Gemma-7B).
* **RAM:** 16GB+ System RAM.

## 🛠 My Specs

#### Hardware

* **CPU:** AMD Ryzen 5 7235HS, 3201 Mhz, 4 Core(s), 8 Logical Processor(s).
* **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM).
* **RAM:** 16GB System RAM.

### Software

* **Anaconda** or Miniconda installed.
* **Hugging Face Account:** You must accept the license terms for Gemma on Hugging Face and generate an Access Token.

---

## 📦 Installation

1. **Create the Anaconda Environment**

```bash
conda create -n gemma-mtg python=3.12
conda activate gemma-mtg
```

1. **Install PyTorch (CUDA support)**
*Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the specific command for your CUDA version. Example:*

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

1. **Install Dependencies**

```bash
pip install transformers datasets accelerate bitsandbytes peft sentencepiece protobuf trl wandb mtgsdk requests beautifulsoup4 pandas
```

1. **Hugging Face Login**
The Gemma model is gated. You must log in via CLI.

```bash
huggingface-cli login
# Paste your Read-access token when prompted
```

---

## 🏃‍♂️ Workflow & Usage

Follow these scripts in order to build and use your model.

### 1. Data Collection (`collect_combo_data.py`)

This script is the engine of the project. It fetches cards from the Scryfall API and identifies specific mechanics (Storm, Flicker, Untap). It also generates the "reasoning" dataset used for training.

* **Output:** Creates `data/pauper_cards_detailed.json` and `data/combo_training_data.json`.
* **Run:**

```bash
python collect_combo_data.py
```

### 2. Model Training (`train_gemma.py`)

This script fine-tunes `google/gemma-2b-it`. It uses **QLoRA** (Quantized Low-Rank Adaptation) to train efficiently on consumer hardware.

* **Configuration:**
* Base Model: `google/gemma-2b-it`
* Quantization: 4-bit (NF4)
* Epochs: 5

* **Run:**

```bash
python train_gemma.py
```

* *Note: Training will take 1-3 hours depending on your GPU.*

### 3. Interactive Mode (`combo_explorer.py`)

Once trained, use this script to chat with your model. You can ask it to analyze specific cards or suggest pieces.

* **Commands:**
* `combo Card A, Card B`: Asks the model to analyze the interaction.
* `suggest Card Name`: Asks the model what fits well with a specific card.

* **Run:**

```bash
python combo_explorer.py
```

### 4. Automated Discovery (`discover_combos.py`)

This script systematically iterates through high-potential cards (cards with 2+ relevant abilities) and asks the model: *"Do these cards create an infinite loop?"*

* **Function:** It filters for "combo-potential" cards and runs the model against pairs/triplets to find novel interactions.
* **Output:** Saves findings to `data/discovered_combos.json`.
* **Run:**

```bash
python discover_combos.py
```

---

## 📂 File Structure

| File | Description |
| --- | --- |
| `collect_mtg_data.py` | Basic scraper for general deck-building data. |
| `collect_combo_data.py` | **Core Scraper.** Extracts mechanics and generates reasoning datasets. |
| `train_gemma.py` | **Training Script.** Fine-tunes the model using Peft/LoRA. |
| `combo_explorer.py` | Interactive CLI to chat with the trained model. |
| `discover_combos.py` | Automated bot that hunts for new combos in background. |
| `test_model.py` | Simple sanity check to ensure the model loads correctly. |
| `data/` | Directory containing JSON datasets and model outputs. |
| `gemma-mtg-combo-finder/` | Directory where the fine-tuned model weights are saved. |

---

## 🧠 How It Works

### The "Reasoning" Dataset

Standard LLMs struggle with Magic combos because they require strict logic. This project solves that by programmatically generating training examples in `collect_combo_data.py`.

Instead of just feeding card text, we generate examples like this:

> **User:** Do "Midnight Guard" and "Presence of Gond" combo?
> **Model:** Yes.
>
> 1. Presence of Gond grants the ability to tap and create an Elf.
> 2. Midnight Guard has an ability: "Untap when another creature enters."
> 3. Tapping Guard creates an Elf -> Elf enters -> Guard Untaps.
> 4. Result: Infinite Tokens.
>
>

By training on this structure, the model learns to look for the **Trigger -> Action -> Loop** pattern in cards it has never seen combined before.

---

## ⚖️ License & Acknowledgements

* **Card Data:** Courtesy of [Scryfall API](https://scryfall.com/docs/api).
* **Model:** Based on Google DeepMind's [Gemma](https://ai.google.dev/gemma).
* **Note:** This project is unofficial Fan Content permitted under the Fan Content Policy. Not approved/endorsed by Wizards.
