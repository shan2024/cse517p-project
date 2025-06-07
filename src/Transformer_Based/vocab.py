import string
import json


CONTROL_CHARS = ['\t', '\n', '\r', '\v', '\f']

def spanish_vocab():
     return ['á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ']

def build_vocab():
    """Create a vocabulary of printable lowercase English and Spanish characters."""
    
    # Start with ASCII printable characters but filter out uppercase and control chars
    printable_chars = [char for char in string.printable 
                      if not char.isupper() 
                      and char not in string.punctuation and char not in CONTROL_CHARS]
    
    # Combine and sort all characters
    all_chars = sorted(set(printable_chars + spanish_vocab()))
    
    # Create the mappings
    char_to_index = {char: idx for idx, char in enumerate(all_chars)}
    
    return char_to_index

def init_vocab(vocab_file_path):
    """Create a new vocab and write it to the provided folder"""

    vocab = build_vocab()

    with open(vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
            
    return vocab