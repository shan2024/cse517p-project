"""
Helper module for Arabic and Devanagari character sets.
Contains character ranges and common characters for both scripts.
"""
import string
import os
import json

def load_script_vocab(script_name):
    """Load vocabulary from a pre-generated vocab file.
    
    Args:
        script_name: Name of the script (e.g., 'arabic', 'devanagari')
    
    Returns:
        List of characters
        
    Raises:
        FileNotFoundError: If the vocabulary file doesn't exist
        ValueError: If the vocabulary file doesn't contain a valid vocabulary
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    vocab_file = os.path.join(project_root, 'data/vocab_files', f'{script_name}_vocab.json')
    
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"{script_name} vocabulary file not found at {vocab_file}")
    
    try:
        print(f"Loading {script_name} vocabulary from {vocab_file}")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            if 'vocabulary' not in vocab_data:
                raise ValueError(f"Invalid vocabulary file format: 'vocabulary' key not found in {vocab_file}")
            chars = vocab_data['vocabulary']
        print(f"Loaded {len(chars)} {script_name} characters")
        return chars
    except Exception as e:
        raise RuntimeError(f"Error loading {script_name} vocabulary file: {e}")

def get_arabic_char_ranges():
    """Return Unicode ranges for Arabic script characters"""
    return [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0x0870, 0x089F),  # Arabic Extended-B
        # Removing presentation forms which create duplicates
        # (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        # (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        (0x10E60, 0x10E7F),  # Rumi Numeral Symbols
        # Removing mathematical symbols rarely used in text
        # (0x1EE00, 0x1EEFF)  # Arabic Mathematical Alphabetic Symbols
    ]

def get_devanagari_char_ranges():
    """Return Unicode ranges for Devanagari script characters used by Hindi, Marathi, and Nepali"""
    return [
        (0x0900, 0x097F),  # Devanagari (standard range for Hindi, Marathi, Nepali)
        # Remove Devanagari Extended which contains some specialized symbols
        # Remove Vedic Extensions (0x1CD0-0x1CFF) which are rarely used in modern text
    ]

def build_arabic_charset():
    """Build a set of common Arabic script characters"""
    # Core Arabic characters (including basic Farsi, Urdu additions)
    chars = []
    
    # Add the most common Arabic characters directly
    base_arabic = [
        # Arabic letters
        'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي',
        # Hamza variants
        'ء', 'أ', 'إ', 'آ', 'ؤ', 'ئ',
        # Arabic diacritics
        'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٌ', 'ٍ', 'ً',
        # Arabic-Indic digits
        '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩',
        # Extended Arabic for Farsi
        'پ', 'چ', 'ژ', 'گ',
        # Extended Arabic for Urdu
        'ٹ', 'ڈ', 'ڑ', 'ں', 'ے',
        # Common Arabic symbols and punctuation
        '،', '؟', '؛'
    ]
    
    # Add base_arabic first
    chars.extend(base_arabic)
    
    # Selectively add from defined ranges, only characters not already included
    added_from_ranges = set()
    for start, end in get_arabic_char_ranges():
        for codepoint in range(start, end + 1):
            # Only add printable characters that aren't presentation forms
            char = chr(codepoint)
            if (char.isprintable() and 
                char not in chars and 
                char not in added_from_ranges and
                # Skip presentation forms (these are duplicates of base forms)
                not (0xFB50 <= codepoint <= 0xFDFF) and 
                not (0xFE70 <= codepoint <= 0xFEFF)):
                
                # Limit to reasonable character count
                if len(chars) < 250:
                    chars.append(char)
                    added_from_ranges.add(char)
    
    # Return a list with no more than 256 characters
    return chars[:256]

def build_devanagari_charset():
    """Build a set of common Devanagari script characters for Hindi, Marathi, and Nepali"""
    chars = []
    
    # Define Devanagari punctuation and special characters to exclude
    devanagari_exclude = {
        # Punctuation
        '।',  # Devanagari Danda (full stop)
        '॥',  # Devanagari Double Danda (paragraph/section end)
        '॰',  # Devanagari Abbreviation Sign
        'ऽ',  # Devanagari Avagraha
        
        # Exclude any Vedic extension characters (Unicode range 0x1CD0-0x1CFF)
        # These characters typically start with '꣠', '꣡', etc.
    }
    
    # Add characters from defined ranges
    for start, end in get_devanagari_char_ranges():
        for codepoint in range(start, end + 1):
            # Only add printable characters that aren't in the exclude list
            char = chr(codepoint)
            if char.isprintable() and char not in devanagari_exclude:
                # Skip characters that look like punctuation or decorative marks
                if not char in string.punctuation and not (0x1CD0 <= ord(char) <= 0x1CFF):
                    chars.append(char)
    
    # Add core Devanagari characters directly (for Hindi, Marathi, and Nepali)
    base_devanagari = [
        # Vowels
        'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ',
        # Consonants
        'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',
        'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
        'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह',
        # Matras (vowel signs)
        'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ', '्',
        # Digits
        '०', '१', '२', '३', '४', '५', '६', '७', '८', '९',
        # Additional characters for Hindi, Marathi, and Nepali
        'क्ष', 'त्र', 'ज्ञ', 'ऋ', 'ॠ', 'ऌ', 'ॡ', 'ऑ', 'ॉ'
    ]
    
    # Add unique chars from base_devanagari that might not be in the ranges
    for char in base_devanagari:
        if char not in chars and char not in devanagari_exclude:
            chars.append(char)
    
    return chars

def arabic_vocab():
    """Return a list of common Arabic script characters from the vocabulary file."""
    return load_script_vocab('arabic')

def devanagari_vocab():
    """Return a list of common Devanagari script characters from the vocabulary file."""
    return load_script_vocab('devanagari')
