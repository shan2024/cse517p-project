# Count frequency of all charactres in cjk language family
# Display number of unque chars for each language 
# Build a vocab list that contains 95% of the probility mass of characters for each language

import os
import glob
import collections
import argparse
import json
from typing import Dict, List, Tuple


def parse_arguments():
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    
    parser = argparse.ArgumentParser(description='Process CJK language text to build vocabulary lists')
    parser.add_argument('--data_dir', type=str, default=os.path.join(project_root, 'data/parsed_data'),
                        help='Directory containing language data files')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'data'),
                        help='Directory to save output vocabulary files')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Probability mass threshold for vocabulary')
    return parser.parse_args()


def get_files_by_language(data_dir: str) -> Dict[str, List[str]]:
    """Get text files organized by language."""
    # Check if data_dir exists
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist")
        os.makedirs(data_dir, exist_ok=True)
    
    # Check if cjk directory exists
    cjk_dir = os.path.join(data_dir, 'arabic')
    if not os.path.exists(cjk_dir):
        print(f"Warning: arabic directory {cjk_dir} does not exist")
        os.makedirs(cjk_dir, exist_ok=True)
    
    # Define pattern for language files (CSV files)
    language_files = {
        'Arabic': glob.glob(os.path.join(data_dir, 'arabic', '*_ar.csv')),
        'Farsi': glob.glob(os.path.join(data_dir, 'arabic', '*_fa.csv')),
        'Urdu': glob.glob(os.path.join(data_dir, 'arabic', '*_ur.csv')),
        'Pashto': glob.glob(os.path.join(data_dir, 'arabic', '*_ps.csv')),
        'Uyghur': glob.glob(os.path.join(data_dir, 'arabic', '*_ug.csv')),
        'Kurdish': glob.glob(os.path.join(data_dir, 'arabic', '*_ku.csv')),
    }
    return language_files


def count_characters(files: List[str]) -> collections.Counter:
    """Count character frequencies in a list of files."""
    counter = collections.Counter()
    for file_path in files:
        try:
            if file_path.endswith('.csv'):
                # For CSV files, read line by line and extract text content
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Skip header if present
                    header = f.readline()
                    for line in f:
                        # Assuming the text content is in the second column (index 1)
                        # Adjust this if the CSV structure is different
                        try:
                            parts = line.strip().split(',')
                            if len(parts) > 1:
                                text = parts[1]  # Adjust index based on CSV structure
                                counter.update(text)
                        except Exception as e:
                            print(f"Error processing line in {file_path}: {e}")
            else:
                # For regular text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    counter.update(text)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return counter


def build_vocab(counter: collections.Counter, threshold: float) -> Tuple[List[str], float]:
    """Build a vocabulary that covers the given probability mass threshold."""
    total_count = sum(counter.values())
    
    # Sort characters by frequency
    sorted_chars = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    vocab = []
    cumulative_prob = 0.0
    
    for char, count in sorted_chars:
        if char.isspace() or not char.isprintable():
            continue
            
        prob = count / total_count
        cumulative_prob += prob
        vocab.append(char)
        
        if cumulative_prob >= threshold:
            break
    
    coverage = cumulative_prob
    return vocab, coverage


def process_language(language: str, files: List[str], threshold: float) -> Dict:
    """Process a language and return vocabulary statistics."""
    print(f"Processing {language} language...")
    
    # Count character frequencies
    counter = count_characters(files)
    
    # Remove non-printable characters and whitespace
    for char in list(counter.keys()):
        if char.isspace() or not char.isprintable():
            del counter[char]
    
    # Get unique character count
    unique_chars = len(counter)
    
    # Build vocabulary
    vocab, coverage = build_vocab(counter, threshold)
    
    # Calculate statistics
    vocab_size = len(vocab)
    vocab_percentage = (vocab_size / unique_chars) * 100 if unique_chars > 0 else 0
    
    return {
        'language': language,
        'unique_characters': unique_chars,
        'vocabulary_size': vocab_size,
        'vocabulary_percentage': vocab_percentage,
        'coverage': coverage,
        'vocabulary': vocab
    }


def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get files by language
    language_files = get_files_by_language(args.data_dir)
    
    # Process each language
    results = {}
    for language, files in language_files.items():
        if not files:
            print(f"No files found for {language}")
            continue
            
        language_result = process_language(language, files, args.threshold)
        results[language] = language_result
        
        # Save vocabulary to file
        output_file = os.path.join(args.output_dir, f'{language}_vocab.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(language_result, f, ensure_ascii=False, indent=2)
    
    # Display summary
    print("\nSummary:")
    print("=" * 60)
    print(f"{'Language':<10} {'Unique Chars':<15} {'Vocab Size':<15} {'Coverage':<10}")
    print("-" * 60)
    
    for language, result in results.items():
        print(f"{language:<10} {result['unique_characters']:<15} {result['vocabulary_size']:<15} {result['coverage']:.4f}")
    
    print("=" * 60)
    print(f"Vocabulary files saved to {args.output_dir}")


if __name__ == "__main__":
    main()