#!/usr/bin/env python
"""
Test script to verify the Arabic and Devanagari character support.
This script will create a sample vocabulary with the new character sets
and print statistics about the vocabulary.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to the path so we can import from Transformer_Based
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Transformer_Based.vocab import init_vocab, build_vocab
from Transformer_Based.arabic_devanagari_support import arabic_vocab, devanagari_vocab

def test_vocab_support():
    """Test the Arabic and Devanagari character support in the vocabulary."""
    print("Testing Arabic and Devanagari character support...")
    
    # Get counts of characters in each script
    arabic_chars = arabic_vocab()
    devanagari_chars = devanagari_vocab()
    
    print(f"Number of Arabic script characters: {len(arabic_chars)}")
    print(f"Number of Devanagari script characters: {len(devanagari_chars)}")
    
    # Sample characters from each script
    print("\nSample Arabic characters:")
    print("".join(arabic_chars[:20]))
    
    print("\nSample Devanagari characters:")
    print("".join(devanagari_chars[:20]))
    
    # Build a complete vocabulary
    print("\nBuilding complete vocabulary...")
    vocab = build_vocab("all")
    print(f"Total vocabulary size: {len(vocab)}")
    
    # Check if Arabic and Devanagari characters are in the vocabulary
    arabic_in_vocab = sum(1 for char in arabic_chars if char in vocab)
    devanagari_in_vocab = sum(1 for char in devanagari_chars if char in vocab)
    
    print(f"Arabic characters in vocabulary: {arabic_in_vocab}/{len(arabic_chars)}")
    print(f"Devanagari characters in vocabulary: {devanagari_in_vocab}/{len(devanagari_chars)}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_vocab_support()
