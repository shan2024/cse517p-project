#!/usr/bin/env python3

import argparse
import os
import torch
from tabulate import tabulate
# Import helper functions
from helpers import load_true, load_predicted, get_top1_accuracy, get_top3_accuracy, load_test_input, write_pred
# Import model wrapper directly
from Transformer_Based.transformer_wrapper import TransformerModelWrapper
import time

# Define language name mappings
LANGUAGE_NAMES = {
    # Germanic languages
    "en": "English",
    "de": "German",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "is": "Icelandic",
    "af": "Afrikaans",
    "fy": "Frisian",
    "nds": "Low German",
    "als": "Alemannic German",
    "bar": "Bavarian",
    "frr": "North Frisian",
    "vls": "West Flemish",
    
    # Romance languages
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ca": "Catalan",
    "gl": "Galician",
    "eu": "Basque",
    "eo": "Esperanto",
    "la": "Latin",
    "oc": "Occitan",
    "ast": "Asturian",
    "pms": "Piedmontese",
    "lmo": "Lombard",
    "an": "Aragonese",
    "ilo": "Ilokano",
    "rm": "Romansh",
    "wa": "Walloon",
    "vec": "Venetian",
    "nap": "Neapolitan",
    "scn": "Sicilian",
    "mwl": "Mirandese",
    "pam": "Kapampangan",
    "bcl": "Central Bicolano",
    
    # Slavic languages
    "ru": "Russian",
    "pl": "Polish",
    "cs": "Czech",
    "uk": "Ukrainian",
    "bg": "Bulgarian",
    "sk": "Slovak",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "sl": "Slovenian",
    "et": "Estonian",
    "sr": "Serbian",
    "mk": "Macedonian",
    "be": "Belarusian",
    "hr": "Croatian",
    "bs": "Bosnian",
    "sh": "Serbo-Croatian",
    "hsb": "Upper Sorbian",
    "dsb": "Lower Sorbian",
    "rue": "Rusyn",
    
    # Sino-Tibetan languages
    "zh": "Chinese",
    "my": "Burmese",
    "bo": "Tibetan",
    "wuu": "Wu Chinese",
    "yue": "Cantonese",
    
    # Japonic languages
    "ja": "Japanese",
    
    # Koreanic languages
    "ko": "Korean",
    
    # Afroasiatic languages
    "ar": "Arabic",
    "he": "Hebrew",
    "am": "Amharic",
    "arz": "Egyptian Arabic",
    
    # Turkic languages
    "tr": "Turkish",
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "ug": "Uyghur",
    "uz": "Uzbek",
    "tt": "Tatar",
    "ba": "Bashkir",
    "cv": "Chuvash",
    "sah": "Yakut",
    "tk": "Turkmen",
    "krc": "Karachay-Balkar",
    "tyv": "Tuvan",
    "xal": "Kalmyk",
    
    # Indo-Iranian languages
    "fa": "Persian",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "ur": "Urdu",
    "ne": "Nepali",
    "mr": "Marathi",
    "ml": "Malayalam",
    "te": "Telugu",
    "kn": "Kannada",
    "gu": "Gujarati",
    "si": "Sinhala",
    "pa": "Punjabi",
    "ps": "Pashto",
    "ku": "Kurdish",
    "sd": "Sindhi",
    "or": "Odia",
    "as": "Assamese",
    "dv": "Dhivehi",
    "pnb": "Western Punjabi",
    "mzn": "Mazandarani",
    "lez": "Lezgian",
    "lrc": "Northern Luri",
    "bh": "Bihari",
    "mai": "Maithili",
    
    # Uralic languages
    "hu": "Hungarian",
    "fi": "Finnish",
    "et": "Estonian",
    "kv": "Komi",
    "mhr": "Eastern Mari",
    "mrj": "Western Mari",
    "myv": "Erzya",
    
    # Austronesian languages
    "id": "Indonesian",
    "tl": "Tagalog",
    "ms": "Malay",
    "jv": "Javanese",
    "su": "Sundanese",
    "ceb": "Cebuano",
    "war": "Waray",
    "min": "Minangkabau",
    "mg": "Malagasy",
    
    # Kartvelian languages
    "ka": "Georgian",
    "xmf": "Mingrelian",
    
    # Mongolic languages
    "mn": "Mongolian",
    "bxr": "Buryat",
    
    # Tai-Kadai languages
    "th": "Thai",
    "lo": "Lao",
    
    # Austro-Asiatic languages
    "vi": "Vietnamese",
    "km": "Khmer",
    
    # Celtic languages
    "cy": "Welsh",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "br": "Breton",
    "kw": "Cornish",
    
    # Constructed languages
    "vo": "Volapük",
    "jbo": "Lojban",
    "io": "Ido",
    "ia": "Interlingua",
    "ie": "Interlingue",
    
    # Other languages
    "sq": "Albanian",
    "hy": "Armenian",
    "yi": "Yiddish",
    "mt": "Maltese",
    "sw": "Swahili",
    "so": "Somali",
    "yo": "Yoruba",
    "gn": "Guarani",
    "qu": "Quechua",
    "nah": "Nahuatl",
    "sa": "Sanskrit"
}

# Define linguistic families with character sets
LINGUISTIC_FAMILIES = {
    "Germanic": {
        "languages": ["en", "de", "nl", "sv", "da", "no", "is", "af", "fy", "nds", "als", "bar", "frr", "vls"],
        "character_set": "Latin"
    },
    "Romance": {
        "languages": ["es", "fr", "it", "pt", "ro", "ca", "gl", "eu", "eo", "la", "oc", "ast", "pms", "lmo", "an", "ilo", "rm", "wa", "vec", "nap", "scn", "mwl", "pam", "bcl"],
        "character_set": "Latin"
    },
    "Slavic": {
        "languages": ["ru", "pl", "cs", "uk", "bg", "sk", "lt", "lv", "sl", "et", "sr", "mk", "be", "hr", "bs", "sh", "hsb", "dsb", "rue"],
        "character_set": "Cyrillic/Latin"
    },
    "Sino-Tibetan": {
        "languages": ["zh", "my", "bo", "wuu", "yue"],
        "character_set": "Chinese/Tibetan"
    },
    "Japonic": {
        "languages": ["ja"],
        "character_set": "Hiragana/Katakana/Kanji"
    },
    "Koreanic": {
        "languages": ["ko"],
        "character_set": "Hangul"
    },
    "Afroasiatic": {
        "languages": ["ar", "he", "am", "arz"],
        "character_set": "Arabic/Hebrew/Ethiopic"
    },
    "Turkic": {
        "languages": ["tr", "az", "kk", "ky", "ug", "uz", "tt", "ba", "cv", "sah", "tk", "krc", "tyv", "xal"],
        "character_set": "Cyrillic/Latin"
    },
    "Indo-Iranian": {
        "languages": ["fa", "hi", "bn", "ta", "ur", "ne", "mr", "ml", "te", "kn", "gu", "si", "pa", "ps", "ku", "sd", "or", "as", "dv", "pnb", "mzn", "lez", "lrc", "bh", "mai"],
        "character_set": "Arabic/Devanagari/Various"
    },
    "Uralic": {
        "languages": ["hu", "fi", "et", "kv", "mhr", "mrj", "myv"],
        "character_set": "Cyrillic/Latin"
    },
    "Austronesian": {
        "languages": ["id", "tl", "ms", "jv", "su", "ceb", "war", "min", "mg"],
        "character_set": "Latin"
    },
    "Kartvelian": {
        "languages": ["ka", "xmf"],
        "character_set": "Georgian"
    },
    "Mongolic": {
        "languages": ["mn", "bxr"],
        "character_set": "Cyrillic/Mongolian"
    },
    "Tai-Kadai": {
        "languages": ["th", "lo"],
        "character_set": "Thai/Lao"
    },
    "Austro-Asiatic": {
        "languages": ["vi", "km"],
        "character_set": "Latin/Khmer"
    },
    "Celtic": {
        "languages": ["cy", "ga", "gd", "br", "kw"],
        "character_set": "Latin"
    },
    "Constructed": {
        "languages": ["eo", "vo", "jbo", "io", "ia", "ie"],
        "character_set": "Latin"
    },
    "Other": {
        "languages": ["sq", "hy", "eu", "yi", "mt", "sw", "so", "yo", "gn", "qu", "nah", "sa"],
        "character_set": "Various"
    }
}

def get_language_name(lang_code):
    """Return the full language name for a given language code."""
    if lang_code == "Combined":
        return "Combined"
    return LANGUAGE_NAMES.get(lang_code, lang_code)

def get_language_family(lang_code):
    """Return the linguistic family and character set for a given language code."""
    for family, info in LINGUISTIC_FAMILIES.items():
        if lang_code in info["languages"]:
            return family, info["character_set"]
    return "Unknown", "Unknown"

def calculate_averages(results):
    """Calculate average accuracies by family and character set."""
    family_averages = {}
    charset_data = {}
    
    for family, family_data in results.items():
        if family == "Combined":
            continue
            
        char_set = family_data["character_set"]
        top1_scores = []
        top3_scores = []
        
        for lang_result in family_data["languages"]:
            # Skip error results
            if lang_result[1] != "Error" and lang_result[2] != "Error":
                try:
                    top1_val = float(lang_result[1].replace('%', ''))
                    top3_val = float(lang_result[2].replace('%', ''))
                    top1_scores.append(top1_val)
                    top3_scores.append(top3_val)
                except ValueError:
                    continue
        
        if top1_scores and top3_scores:
            family_avg_top1 = sum(top1_scores) / len(top1_scores)
            family_avg_top3 = sum(top3_scores) / len(top3_scores)
            family_averages[family] = {
                "top1": family_avg_top1,
                "top3": family_avg_top3,
                "char_set": char_set,
                "count": len(top1_scores)
            }
            
            # Aggregate by character set
            if char_set not in charset_data:
                charset_data[char_set] = {"top1_scores": [], "top3_scores": [], "families": []}
            charset_data[char_set]["top1_scores"].extend(top1_scores)
            charset_data[char_set]["top3_scores"].extend(top3_scores)
            charset_data[char_set]["families"].append(family)
    
    # Calculate character set averages
    charset_averages = {}
    for char_set, data in charset_data.items():
        if data["top1_scores"] and data["top3_scores"]:
            charset_averages[char_set] = {
                "top1": sum(data["top1_scores"]) / len(data["top1_scores"]),
                "top3": sum(data["top3_scores"]) / len(data["top3_scores"]),
                "families": list(set(data["families"])),
                "count": len(data["top1_scores"])
            }
    
    return family_averages, charset_averages

def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy by language")
    parser.add_argument("--work_dir", type=str, default="./work", 
                        help="Directory containing the model and vocabulary")
    parser.add_argument("--test_base_dir", type=str, default="./test", 
                        help="Base directory containing test data")
    args = parser.parse_args()
    
    # Dynamically build test directories from subdirectories
    test_dirs = {"Combined": args.test_base_dir}
    
    # Add all subdirectories as separate languages
    for item in os.listdir(args.test_base_dir):
        full_path = os.path.join(args.test_base_dir, item)
        if os.path.isdir(full_path):
            test_dirs[item] = full_path
    
    results = {}  # Group results by family
    
    # Set up device and model once to reuse
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModelWrapper(device,args.work_dir)
    model.load()
    
    # Run evaluation for each language directory
    for language, test_dir in test_dirs.items():
        
        if not os.path.exists(test_dir):
            print(f"Warning: Test directory {test_dir} does not exist. Skipping {language} evaluation.")
            continue
            
        # Get full language name
        language_name = get_language_name(language)
        print(f"\n===== Evaluating {language_name} ({language}) =====")
        
        # Get family and character set info
        family, char_set = get_language_family(language)
        if language == "Combined":
            family, char_set = "Combined", "Mixed"
        
        try:
            # Step 1: Generate predictions directly
            print(f"Generating predictions for {language}...")
            input_file = os.path.join(test_dir, "input.txt")
            pred_file = os.path.join(test_dir, "pred.txt")
            
            # Load test input
            test_input = load_test_input(input_file)

            start = time.perf_counter()
            
            # Generate predictions in batches to avoid memory issues
            batch_size = 5000
            preds = []
            for i in range(0, len(test_input), batch_size):
                batch = test_input[i:i + batch_size]
                preds.extend(model.predict(batch))

            end = time.perf_counter()
            elapsed_time = end - start
            
            # Calculate number of examples and time per prediction
            num_examples = len(test_input)
            ms_per_prediction = (elapsed_time / num_examples) * 1000 if num_examples > 0 else 0
            
            # Write predictions to file
            write_pred(preds, pred_file)

            # Step 2: Direct evaluation using helpers
            print(f"Evaluating accuracy for {language}...")
            
            # Load true labels and predictions
            y_true = load_true(test_dir)
            pred = load_predicted(test_dir)
            
            # Calculate top-1 and top-3 accuracy
            top1_acc = get_top1_accuracy(y_true, pred)
            top3_acc = get_top3_accuracy(y_true, pred)
            top1_pct = top1_acc * 100
            top3_pct = top3_acc * 100
            
            # Initialize family group if not exists
            if family not in results:
                results[family] = {
                    "character_set": char_set,
                    "languages": []
                }
            # Store results grouped by family with full language name
            results[family]["languages"].append([
                f"{language_name} ({language})", 
                f"{top1_pct:.2f}%", 
                f"{top3_pct:.2f}%", 
                f"{elapsed_time:.2f} sec",
                f"{num_examples}",
                f"{ms_per_prediction:.2f} ms"
            ])
            
            # Print results
            print(f"Family: {family} ({char_set})")
            print(f"Top-1 Accuracy: {top1_acc:.2%}")
            print(f"Top-3 Accuracy: {top3_acc:.2%}")
            print(f"Number of examples: {num_examples}")
            print(f"Time per prediction: {ms_per_prediction:.2f} ms")
            
        except Exception as e:
            print(f"Error during evaluation for {language_name} ({language}): {e}")
            if family not in results:
                results[family] = {
                    "character_set": char_set,
                    "languages": []
                }
            results[family]["languages"].append([f"{language_name} ({language})", "Error", "Error", "Error", "Error", "Error"])

    # Print summary table grouped by family
    if results:
        print("\n===== SUMMARY BY LINGUISTIC FAMILY =====")
        
        # Calculate averages
        family_averages, charset_averages = calculate_averages(results)
        
        # Sort families for consistent output
        for family in sorted(results.keys()):
            family_data = results[family]
            print(f"\n{family} Family (Character Set: {family_data['character_set']})")
            print("=" * (len(family) + len(family_data['character_set']) + 25))
            
            if family_data["languages"]:
                # Sort languages by top-3 accuracy (low to high)
                sorted_languages = sorted(family_data["languages"], 
                                         key=lambda x: float(x[2].replace('%', '')) if x[2] != "Error" else -1)
                
                print(tabulate(sorted_languages, headers=[
                    "Language", 
                    "Top-1 Accuracy", 
                    "Top-3 Accuracy", 
                    "Total Time", 
                    "Examples", 
                    "ms/prediction"
                ], tablefmt="grid"))
                
                # Show family average if available
                if family in family_averages:
                    avg_data = family_averages[family]
                    print(f"\nFamily Average (n={avg_data['count']}): Top-1: {avg_data['top1']:.2f}%, Top-3: {avg_data['top3']:.2f}%")
            else:
                print("No languages evaluated in this family.")
        
        # Print family averages summary
        if family_averages:
            print("\n===== FAMILY AVERAGES SUMMARY =====")
            family_summary = []
            for family, avg_data in sorted(family_averages.items()):
                family_summary.append([
                    family,
                    avg_data['char_set'],
                    f"{avg_data['top1']:.2f}%",
                    f"{avg_data['top3']:.2f}%",
                    avg_data['count']
                ])
            
            print(tabulate(family_summary, headers=[
                "Family", 
                "Character Set", 
                "Avg Top-1", 
                "Avg Top-3", 
                "Languages Tested"
            ], tablefmt="grid"))
        
        # Print character set averages summary
        if charset_averages:
            print("\n===== CHARACTER SET AVERAGES SUMMARY =====")
            charset_summary = []
            for char_set, avg_data in sorted(charset_averages.items()):
                families_str = ", ".join(sorted(avg_data['families']))
                charset_summary.append([
                    char_set,
                    f"{avg_data['top1']:.2f}%",
                    f"{avg_data['top3']:.2f}%",
                    avg_data['count'],
                    families_str
                ])
            
            print(tabulate(charset_summary, headers=[
                "Character Set", 
                "Avg Top-1", 
                "Avg Top-3", 
                "Languages Tested",
                "Families"
            ], tablefmt="grid"))
    else:
        print("\nNo results were generated.")

if __name__ == "__main__":
    main()
