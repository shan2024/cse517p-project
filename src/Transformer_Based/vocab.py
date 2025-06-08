import string
import json


CONTROL_CHARS = ['\t', '\n', '\r', '\v', '\f']


def build_vocab(charset: str):
    """Create a vocabulary of characters based on the specified charset."""
    
    # Start with ASCII printable characters but filter out uppercase and control chars
    printable_chars = set([char for char in string.printable 
                      if not char.isupper() 
                      and char not in string.punctuation and char not in CONTROL_CHARS])
    
    all_chars = set.union(printable_chars, build_cyrillic_charset(), build_latin_charset(), build_cjk_charset())
    # if charset == "latin":
    #     all_chars = sorted(build_latin_charset())
    # elif charset == "cyrillic":
    #     cyrillic_chars = build_cyrillic_charset()
    #     all_chars = sorted(set(printable_chars + list(cyrillic_chars)))
    # elif charset == "cjk":
    #     cjk_chars = build_cjk_charset()
    #     all_chars = sorted(set(printable_chars + list(cjk_chars)))
    # else:
    #     raise ValueError(f"Unsupported charset: {charset}")
    
    # Create the mappings
    char_to_index = {char: idx for idx, char in enumerate(all_chars)}
    
    return char_to_index

def init_vocab(vocab_file_path, charset="latin"):
    """Create a new vocab and write it to the provided folder"""

    vocab = build_vocab(charset)

    with open(vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
            
    return vocab

### Define script specific charsets
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
    
    # Remove duplicates by converting to set, then back to list
    unique_special_chars = list(set(all_special_chars))
    
    return set(printable_chars + unique_special_chars)


def build_cyrillic_charset():
    # Include basic ASCII for mixed content support
    printable_chars = [char for char in string.printable 
                      if not char.isupper() 
                      and char not in string.punctuation and char not in CONTROL_CHARS]
    
    # Combine all Cyrillic language-specific characters
    all_cyrillic_chars = (russian_vocab() + ukrainian_vocab() + 
                         bulgarian_vocab() + serbian_vocab() +
                         macedonian_vocab() + belarusian_vocab())
    
    # Remove duplicates by converting to set, then back to list
    unique_cyrillic_chars = list(set(all_cyrillic_chars))
    
    return set(printable_chars + unique_cyrillic_chars)

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
