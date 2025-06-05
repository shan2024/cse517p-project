from datasets import load_dataset
import os
import sys
import csv

# This script downloads the CulturaX dataset in multiple languages, samples a specified number of lines from each language,
# and merges them into a single CSV file for training purposes.
# You need to follow these steps:
# source .venv/bin/activate
# pip install huggingface_hub
# huggingface-cli login
# Access Token: Will be provided
# For more languages, you can visit https://huggingface.co/datasets/uonlp/CulturaX and add them here in the list
# Selecting target languages here. TODO: Add more languages if needed.
langs = ["es", "cmn"] 
lines_per_lang = 5000 # Number of lines to sample per language
output_dir = "data/parsed_data"

# Check to ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Prepare separate writers for train/dev/test splits
split_filenames = {
    "train": os.path.join(output_dir, "train_culturax_multilingual.csv"),
    "dev": os.path.join(output_dir, "dev_culturax_multilingual.csv"),
    "test": os.path.join(output_dir, "test_culturax_multilingual.csv"),
}
split_writers = {}
split_files = {}

for split, path in split_filenames.items():
    f = open(path, "w", encoding="utf-8", newline="")
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["dialogue"])
    split_files[split] = f
    split_writers[split] = writer

# Write multilingual data in three separate files
# Not downlading all the data at once as it is time consuming. Lets stream the data instead based on the language and sample the required number of lines.
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
                    split_writers["train"].writerow([text])
                elif count < int(lines_per_lang * 0.9):
                    split_writers["dev"].writerow([text])
                elif count < lines_per_lang:
                    split_writers["test"].writerow([text])
                else:
                    break
                count += 1

        print(f"Collected {count} lines for {lang}")

    print("multilingual train/dev/test files created and saved:")
    for split, path in split_filenames.items():
        print(f" - {split}: {path}")

except Exception as e:
    print(f"An error occurred while processing the dataset: {e}")
    sys.exit(1)

finally:
    for f in split_files.values():
        f.close()

os._exit(0)
