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
# Selecting target languages here. TODO: Add more languages if needed.
langs = ["es", "zh"] 
lines_per_lang = 10000 # Number of lines to sample per language
output_dir = "data/parsed_data"

# Check to ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Prepare separate writers for train/dev/test splits for each language
lang_split_writers = {}
lang_split_files = {}

# Initialize writers for each language and split combination
for lang in langs:
    lang_split_writers[lang] = {}
    lang_split_files[lang] = {}
    
    for split in ["train", "dev", "test"]:
        filename = os.path.join(output_dir, f"{split}_culturax_{lang}.csv")
        f = open(filename, "w", encoding="utf-8", newline="")
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["dialogue"])
        lang_split_files[lang][split] = f
        lang_split_writers[lang][split] = writer

# Write each language's data to separate files
# Not downloading all the data at once as it is time consuming. Lets stream the data instead based on the language and sample the required number of lines.
try:
    for lang in langs:
        print(f"Streaming CulturaX: {lang}")
        dataset = load_dataset("uonlp/CulturaX", lang, streaming=True)
        stream = dataset["train"]
        
        count = 0
        for example in stream:
            text = example.get("text", "").strip()
            if text:
                text = text.replace("\n", " ")
                if count < int(lines_per_lang * 0.8):
                    lang_split_writers[lang]["train"].writerow([text])
                elif count < int(lines_per_lang * 0.9):
                    lang_split_writers[lang]["dev"].writerow([text])
                elif count < lines_per_lang:
                    lang_split_writers[lang]["test"].writerow([text])
                else:
                    break
                count += 1

        print(f"Collected {count} lines for {lang}")

    print("Language-specific train/dev/test files created and saved:")
    for lang in langs:
        for split in ["train", "dev", "test"]:
            filename = os.path.join(output_dir, f"{split}_culturax_{lang}.csv")
            print(f" - {lang} {split}: {filename}")

except Exception as e:
    print(f"An error occurred while processing the dataset: {e}")
    sys.exit(1)

finally:
    for lang in langs:
        for split in ["train", "dev", "test"]:
            if lang in lang_split_files and split in lang_split_files[lang]:
                lang_split_files[lang][split].close()

os._exit(0)
