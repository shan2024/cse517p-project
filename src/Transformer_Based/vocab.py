import string
import json
import collections
import os
import importlib.util
import subprocess
import sys
from .arabic_devanagari_support import arabic_vocab, devanagari_vocab

CONTROL_CHARS = ['\t', '\n', '\r', '\v', '\f']

def build_vocab(charset="all"):
    """Create a vocabulary of characters based on the specified charset.
    
    Args:
        charset: Base character set to include. Can be:
                - A single charset name: 'latin', 'cyrillic', 'cjk', 'arabic', 'devanagari', or 'all'
                - A list of charset names: ['latin', 'devanagari']
    
    Returns:
        Dictionary mapping characters to indices
    """
    # Always include ASCII lowercase and digits as base vocabulary
    base_chars = set(string.ascii_lowercase + string.digits + ' ')
    
    # No corpus provided, use predefined character sets
    all_chars = base_chars
    
    # Convert string charset to list for consistent processing
    if isinstance(charset, str):
        charsets = [charset]
    else:
        charsets = charset
    
    for cs in charsets:
        if cs == "latin" or cs == "all":
            print("Adding Latin characters")
            all_chars.update(build_latin_charset())
        if cs == "cyrillic" or cs == "all":
            print("Adding Cyrillic characters")
            all_chars.update(build_cyrillic_charset())
            
        if cs == "cjk" or cs == "all":
            print("Adding CJK characters")
            all_chars.update(get_common_cjk_chars(2000))
        
        if cs == "arabic" or cs == "all":
            print("Adding Arabic script characters")
            all_chars.update(arabic_vocab())
            
        if cs == "devanagari" or cs == "all":
            print("Adding Devanagari script characters")
            all_chars.update(devanagari_vocab())
    
    # Create the mappings
    char_to_index = {char: idx for idx, char in enumerate(sorted(all_chars))}
    
    print(f"Created vocabulary with {len(char_to_index)} characters")
    return char_to_index

def get_common_cjk_chars(limit=1000):
    """Return all CJK characters from pre-generated vocabulary files without size limits"""
    common_chars = []
    
    # Load Chinese, Japanese and Korean vocabulary from the vocabulary files
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    vocab_files_dir = os.path.join(project_root, 'data/vocab_files')
    
    # List of languages to load
    languages = ['chinese', 'japanese', 'korean']
    
    # Load all characters from each language vocab file
    for language in languages:
        vocab_file = os.path.join(vocab_files_dir, f'{language}_vocab.json')
        lang_chars = []
        
        if os.path.exists(vocab_file):
            try:
                print(f"Loading all {language} characters from {vocab_file}")
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                    if 'vocabulary' in vocab_data:
                        # Load all characters without any limit
                        lang_chars = vocab_data['vocabulary']
                print(f"Loaded {len(lang_chars)} {language} characters from file")
            except Exception as e:
                print(f"Error loading {language} vocabulary file: {e}")
        else:
            print(f"{language} vocabulary file not found at {vocab_file}")
            
            # Fallback for each language if file not found
            if language == 'japanese':
                # For Japanese, Hiragana and Katakana are essential
                for char_range in [(0x3040, 0x309F), (0x30A0, 0x30FF)]:  # Hiragana, Katakana
                    start, end = char_range
                    for codepoint in range(start, end + 1):
                        char = chr(codepoint)
                        if char not in common_chars and char not in lang_chars:
                            lang_chars.append(char)
            elif language == 'korean':
                try:
                    if importlib.util.find_spec("hgtk") is None:
                        print("Installing hgtk package for Korean character handling...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "hgtk"])
                    
                    import hgtk
                    # Add basic Korean Hangul characters
                    for char in list(hgtk.letter.CHO) + list(hgtk.letter.JOONG) + list(hgtk.letter.JONG):
                        if char not in common_chars and char not in lang_chars:
                            lang_chars.append(char)
                except Exception as e:
                    print(f"Could not use hgtk for Korean: {e}")
                    # Add some common Korean Hangul syllables as fallback
                    for codepoint in range(0xAC00, 0xAC00 + 500):  # Add a reasonable number as fallback
                        char = chr(codepoint)
                        if char not in common_chars and char not in lang_chars:
                            lang_chars.append(char)
            elif language == 'chinese':
                # Fallback for Chinese - use common CJK Unified Ideographs
                for codepoint in range(0x4E00, 0x4E00 + 2000):  # Add a reasonable number as fallback
                    char = chr(codepoint)
                    if char not in common_chars and char not in lang_chars:
                        lang_chars.append(char)
        
        print(f"Added {len(lang_chars)} {language} characters")
        common_chars.extend(lang_chars)
    
    # Remove duplicates but preserve order without limiting size
    result = list(dict.fromkeys(common_chars))  # Use dict.fromkeys to preserve order while removing duplicates
    print(f"Final CJK vocabulary size: {len(result)}")
    return result

def parse_top_chinese_chars(filepath, limit=1000):
    """Parse the top Chinese characters file and return the characters."""
    chars = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # The character should be in the second column
                    char = parts[1]
                    if len(char) == 1 and char.isprintable():
                        chars.append(char)
                        if len(chars) >= limit:
                            break
    except Exception as e:
        print(f"Error parsing top Chinese characters file: {e}")
    
    return chars

def init_vocab(vocab_file_path, charset="all"):
    """Create a new vocab and write it to the provided folder"""

    vocab = build_vocab(charset)

    with open(vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
            
    return vocab


def get_common_cyrillic_chars():
    """Return the most common Cyrillic script characters"""
    # Russian and Ukrainian are the most widely used Cyrillic scripts
    return russian_vocab() + ukrainian_vocab()


# Define script specific charsets
def build_latin_charset():
    printable_chars = [char for char in string.printable 
                      if not char.isupper() 
                      and char not in string.punctuation and char not in CONTROL_CHARS]
    
    # Combine all language-specific characters
    all_special_chars = (spanish_vocab() + french_vocab() + 
                        italian_vocab() + portuguese_vocab() +
                        german_vocab() + dutch_vocab() + 
                        swedish_vocab() + norwegian_vocab() +
                        filipino_vocab() + swahili_vocab() + 
                        uzbek_vocab())
    
    return set(all_special_chars)


def build_cyrillic_charset():
    
    # Combine all Cyrillic language-specific characters
    all_cyrillic_chars = (russian_vocab() + ukrainian_vocab() + 
                         bulgarian_vocab() + serbian_vocab() +
                         macedonian_vocab() + belarusian_vocab())
    
    
    return set(all_cyrillic_chars)

def build_cjk_charset():
    # Core blocks containing the most common and printable CJK characters
    cjk_ranges = [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Extension A (fairly common)
        (0x3040, 0x309F),   # Hiragana
        (0x30A0, 0x30FF),   # Katakana
        (0x31F0, 0x31FF),   # Katakana Phonetic Extensions
        (0xAC00, 0xD7AF),   # Hangul Syllables
        (0x1100, 0x11FF),   # Hangul Jamo (basic components)
        (0x3130, 0x318F),   # Hangul Compatibility Jamo
    ]

    charset = set()
    for start, end in cjk_ranges:
        for codepoint in range(start, end + 1):
            ch = chr(codepoint)
            if ch.isprintable():
                charset.add(ch)

    return charset


# Define charsets for latin char languages
def spanish_vocab():
     return ['á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ']

def french_vocab():
    return ['à', 'â', 'ä', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ö', 'ù', 'û', 'ü', 'ÿ', 'œ']

def italian_vocab():
    return ['à', 'è', 'é', 'ì', 'í', 'î', 'ò', 'ó', 'ù', 'ú']

def portuguese_vocab():
    return ['á', 'à', 'â', 'ã', 'ä', 'ç', 'é', 'ê', 'í', 'ó', 'ô', 'õ', 'ú', 'ü']

def german_vocab():
    return ['ä', 'ö', 'ü', 'ß']

def dutch_vocab():
    return ['á', 'à', 'é', 'è', 'í', 'ì', 'ó', 'ò', 'ú', 'ù', 'ë', 'ï', 'ö', 'ü', 'ç', 'ñ', 'ê', 'ô', 'û', 'â', 'î', 'ä']

def swedish_vocab():
    return ['å', 'ä', 'ö', 'é', 'ü']

def norwegian_vocab():
    return ['æ', 'ø', 'å', 'é', 'è', 'ê', 'ó', 'ò', 'ô', 'à', 'á', 'ü']

def filipino_vocab():
    return ['á', 'à', 'â', 'é', 'è', 'ê', 'í', 'ì', 'î', 'ó', 'ò', 'ô', 'ú', 'ù', 'û', 'ñ']

def swahili_vocab():
    return ['á', 'é', 'í', 'ó', 'ú', 'ñ']

def uzbek_vocab():
    return ['ʻ', 'gʻ', 'oʻ', 'ş', 'ç', 'aʻ', 'eʻ', 'iʻ', 'uʻ']

# Define charsets for cyrillic languages
def russian_vocab():
    return ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']

def ukrainian_vocab():
    return ['а', 'б', 'в', 'г', 'ґ', 'д', 'е', 'є', 'ж', 'з', 'и', 'і', 'ї', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я']

def bulgarian_vocab():
    return ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ь', 'ю', 'я']

def serbian_vocab():
    return ['а', 'б', 'в', 'г', 'д', 'ђ', 'е', 'ж', 'з', 'и', 'ј', 'к', 'л', 'љ', 'м', 'н', 'њ', 'о', 'п', 'р', 'с', 'т', 'ћ', 'у', 'ф', 'х', 'ц', 'ч', 'џ', 'ш']

def macedonian_vocab():
    return ['а', 'б', 'в', 'г', 'д', 'ѓ', 'е', 'ж', 'з', 'ѕ', 'и', 'ј', 'к', 'л', 'љ', 'м', 'н', 'њ', 'о', 'п', 'р', 'с', 'т', 'ќ', 'у', 'ф', 'х', 'ц', 'ч', 'џ', 'ш']

def belarusian_vocab():
    return ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'і', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ў', 'ф', 'х', 'ц', 'ч', 'ш', 'ы', 'ь', 'э', 'ю', 'я']
