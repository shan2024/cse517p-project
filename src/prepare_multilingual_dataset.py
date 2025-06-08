from datasets import load_dataset
import os
import sys
import csv

# This script downloads the CulturaX dataset in multiple languages, samples a specified number of lines from each language,
# and saves them into separate CSV files for each language for training purposes.
# You need to follow these steps:
# source .venv/bin/activate
# pip install huggingface_hub
# huggingface-cli login
# Access Token: Will be provided
# For more languages, you can visit https://huggingface.co/datasets/uonlp/CulturaX and add them here in the list
# Selecting target languages here.

langs = [

    ######### LATIN CHARSET LANGUAGES ###############
    # Top romance languages
    "es",  # Spanish
    "fr",  # French
    "it",  # Italian
    "pt",  # Portuguese

    # Top germanic languages
    "de", #: "German",
    "nl", # "Dutch",
    "sv", # "Swedish",
    "no", # "Norwegian",

    # Various other common languages with latin charsets
    "tl",  # Filipino
    "sw",  # Swahili
    "uz",  # Uzbek

    ############### Cyrillic languages ################
    "ru",  # Russian
    "uk",  # Ukrainian
    "bg",  # Bulgarian
    "sr",  # Serbian (Cyrillic variant)
    "mk",  # Macedonian
    "be",  # Belarusian
    
    #########################   CJK languages #############################
    "zh",  # Chinese (Simplified and Traditional) – uses Han characters
    "ja",  # Japanese – uses Kanji (Han), Hiragana, Katakana
    "ko",  # Korean – uses Hangul primarily, Hanja (Han) historically


    #########################  Devanagari Languages #######################
    "hi",  # Hindi – 19.6M documents, 16.8B tokens
    "mr",  # Marathi – 2.2M documents, 1.95B tokens
    "ne",  # Nepali – 3.1M documents, 2.06B tokens

    ############################# Arabic script languages #####################
    "ar",  # Arabic – 74M documents, 69.3B tokens
    "fa",  # Persian (Farsi) – 59.5M documents, 45.9B tokens
    "ur",  # Urdu – 2.8M documents, 2.7B tokens
    "ps",  # Pashto – ~40–50 million speakers
    "ug",  # Uyghur – ~10–12 million speakers
    "ku",  # Kurdish (Sorani dialect uses Arabic script) – ~30–35 million speakers (out of ~40M total Kurds)
]

# Define language family mapping
lang_families = {
    # Latin charset languages
    # Romance languages
    "es": "latin/romance",  # Spanish
    "fr": "latin/romance",  # French
    "it": "latin/romance",  # Italian
    "pt": "latin/romance",  # Portuguese
    
    # Germanic languages
    "de": "latin/germanic",  # German
    "nl": "latin/germanic",  # Dutch
    "sv": "latin/germanic",  # Swedish
    "no": "latin/germanic",  # Norwegian
    
    # Other Latin charset languages
    "tl": "latin/other",  # Filipino
    "sw": "latin/other",  # Swahili
    "uz": "latin/other",  # Uzbek
    
    # Cyrillic languages
    "ru": "cyrillic",  # Russian
    "uk": "cyrillic",  # Ukrainian
    "bg": "cyrillic",  # Bulgarian
    "sr": "cyrillic",  # Serbian
    "mk": "cyrillic",  # Macedonian
    "be": "cyrillic",  # Belarusian
    
    # CJK languages
    "zh": "cjk",  # Chinese
    "ja": "cjk",  # Japanese
    "ko": "cjk",  # Korean
    
    # Devanagari languages
    "hi": "devanagari",  # Hindi
    "mr": "devanagari",  # Marathi
    "ne": "devanagari",  # Nepali
    
    # Arabic script languages
    "ar": "arabic",  # Arabic
    "fa": "arabic",  # Persian (Farsi)
    "ur": "arabic",  # Urdu
    "ps": "arabic",  # Pashto
    "ug": "arabic",  # Uyghur
    "ku": "arabic",  # Kurdish (Sorani)
}

# Define number of lines per language, doubling for CJK languages
DEFAULT_LINES = 10000
lines_per_lang_dict = {}

# Set the number of lines for each language based on family
for lang in langs:
    # Double the lines for CJK languages
    if lang_families.get(lang) == "cjk":
        lines_per_lang_dict[lang] = DEFAULT_LINES * 2
    else:
        lines_per_lang_dict[lang] = DEFAULT_LINES

output_dir = "data/parsed_data"

# Check to ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create subdirectories for each language family
for family in set(lang_families.values()):
    family_dir = os.path.join(output_dir, family)
    os.makedirs(family_dir, exist_ok=True)

# Function to check if all split files exist for a language
def files_exist_for_language(lang):
    """Check if train, dev, and test files already exist for a language"""
    family = lang_families.get(lang, "other")
    family_dir = os.path.join(output_dir, family)
    for split in ["train", "dev", "test"]:
        filename = os.path.join(family_dir, f"{split}_culturax_{lang}.csv")
        if not os.path.exists(filename):
            return False
    return True

# Filter languages to only include those that need to be processed
langs_to_process = [lang for lang in langs if not files_exist_for_language(lang)]

if not langs_to_process:
    print("All languages have already been processed. No downloads needed.")
    sys.exit(0)

print(f"Languages to process: {langs_to_process}")
print(f"Skipping already processed languages: {[lang for lang in langs if lang not in langs_to_process]}")

# Prepare separate writers for train/dev/test splits for each language
lang_split_writers = {}
lang_split_files = {}

# Initialize writers for each language and split combination
for lang in langs_to_process:
    lang_split_writers[lang] = {}
    lang_split_files[lang] = {}
    
    family = lang_families.get(lang, "other")
    family_dir = os.path.join(output_dir, family)
    
    for split in ["train", "dev", "test"]:
        filename = os.path.join(family_dir, f"{split}_culturax_{lang}.csv")
        f = open(filename, "w", encoding="utf-8", newline="")
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["dialogue"])
        lang_split_files[lang][split] = f
        lang_split_writers[lang][split] = writer

# Write each language's data to separate files
# Not downloading all the data at once as it is time consuming. Lets stream the data instead based on the language and sample the required number of lines.
try:
    for lang in langs_to_process:
        print(f"Streaming CulturaX: {lang}")
        dataset = load_dataset("uonlp/CulturaX", lang, streaming=True)
        stream = dataset["train"]
        
        count = 0
        lines_for_lang = lines_per_lang_dict[lang]
        for example in stream:
            text = example.get("text", "").strip()
            if text:
                text = text.replace("\n", " ")
                if count < int(lines_for_lang * 0.8):
                    lang_split_writers[lang]["train"].writerow([text])
                elif count < int(lines_for_lang * 0.9):
                    lang_split_writers[lang]["dev"].writerow([text])
                elif count < lines_for_lang:
                    lang_split_writers[lang]["test"].writerow([text])
                else:
                    break
                count += 1

        print(f"Collected {count} lines for {lang} (requested: {lines_for_lang})")
        if count < lines_for_lang:
            print(f"Warning: Only {count} lines available for {lang}, less than requested {lines_for_lang}")

    print("Language-specific train/dev/test files created and saved:")
    for lang in langs_to_process:
        family = lang_families.get(lang, "other")
        family_dir = os.path.join(output_dir, family)
        for split in ["train", "dev", "test"]:
            filename = os.path.join(family_dir, f"{split}_culturax_{lang}.csv")
            print(f" - {lang} {split}: {filename}")

except Exception as e:
    print(f"An error occurred while processing the dataset: {e}")
    sys.exit(1)

finally:
    for lang in langs_to_process:
        for split in ["train", "dev", "test"]:
            if lang in lang_split_files and split in lang_split_files[lang]:
                lang_split_files[lang][split].close()

os._exit(0)
