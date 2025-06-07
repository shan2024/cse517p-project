import string
import json


CONTROL_CHARS = ['\t', '\n', '\r', '\v', '\f']

def spanish_vocab():
     return ['á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ']

def cjk_vocab():
    """
    Returns a string of all CJK (Chinese, Japanese, Korean) characters
    based on their Unicode ranges.
    """
    cjk_ranges = [
        (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
        (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F), # CJK Unified Ideographs Extension C
        (0x2B740, 0x2B81F), # CJK Unified Ideographs Extension D
        (0x2B820, 0x2CEAF), # CJK Unified Ideographs Extension E
        (0x2CEB0, 0x2EBEF), # CJK Unified Ideographs Extension F
        (0x30000, 0x3134F), # CJK Unified Ideographs Extension G
        (0x2F800, 0x2FA1F), # CJK Compatibility Ideographs Supplement
        (0x3040, 0x309F),   # Hiragana (Japanese)
        (0x30A0, 0x30FF),   # Katakana (Japanese)
        (0x31F0, 0x31FF),   # Katakana Phonetic Extensions
        (0xAC00, 0xD7AF),   # Hangul Syllables (Korean)
        (0x1100, 0x11FF),   # Hangul Jamo
        (0x3130, 0x318F),   # Hangul Compatibility Jamo
        (0xA960, 0xA97F),   # Hangul Jamo Extended-A
        (0xD7B0, 0xD7FF),   # Hangul Jamo Extended-B
    ]

    characters = []
    for start, end in cjk_ranges:
        characters.extend(chr(codepoint) for codepoint in range(start, end + 1))

    return characters

def build_vocab():
    """Create a vocabulary of printable lowercase English and Spanish characters."""
    
    # Start with ASCII printable characters but filter out uppercase and control chars
    printable_chars = [char for char in string.printable 
                      if not char.isupper() 
                      and char not in string.punctuation and char not in CONTROL_CHARS]
    
    # Combine and sort all characters
    all_chars = sorted(set(printable_chars + spanish_vocab() + cjk_vocab()))
    
    # Create the mappings
    char_to_index = {char: idx for idx, char in enumerate(all_chars)}
    
    return char_to_index

def init_vocab(vocab_file_path):
    """Create a new vocab and write it to the provided folder"""

    vocab = build_vocab()

    with open(vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
            
    return vocab