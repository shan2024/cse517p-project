import sys
import os
sys.path.append('/home/user/cse517p-project/src')

# Check if the import path is correct
print(f"Python path: {sys.path}")

# Check if the module exists
try:
    from Transformer_Based.vocab import build_vocab
    print("Successfully imported build_vocab")
except ImportError as e:
    print(f"Error importing: {e}")
    
    # List contents of the Transformer_Based directory
    transformer_dir = '/home/user/cse517p-project/src/Transformer_Based'
    if os.path.exists(transformer_dir):
        print(f"Contents of {transformer_dir}:")
        for item in os.listdir(transformer_dir):
            print(f"  - {item}")
    else:
        print(f"Directory {transformer_dir} does not exist")
    
    # Try alternative import
    try:
        import Transformer_Based
        print(f"Transformer_Based module exists at: {Transformer_Based.__file__}")
    except ImportError as e:
        print(f"Cannot import Transformer_Based: {e}")
    
    sys.exit(1)

# Test with only CJK charset
try:
    print("Building CJK vocabulary...")
    vocab = build_vocab(charset="cjk")
    print(f"CJK vocabulary size: {len(vocab)}")
except Exception as e:
    print(f"Error building CJK vocab: {e}")

# Test with all charsets
try:
    print("Building all vocabulary...")
    vocab_all = build_vocab(charset="all")
    print(f"All vocabulary size: {len(vocab_all)}")
except Exception as e:
    print(f"Error building all vocab: {e}")
