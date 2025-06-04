#!/usr/bin/env python3
# filepath: /home/ameli/cse517p-project/evaluate_by_language.py

import argparse
import os
import subprocess
import re
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy by language")
    parser.add_argument("--work_dir", type=str, default="./work", 
                        help="Directory containing the model and vocabulary")
    parser.add_argument("--test_base_dir", type=str, default="./test", 
                        help="Base directory containing test data")
    args = parser.parse_args()
    
    # Define the test directories for each language
    test_dirs = {
        "Combined": args.test_base_dir,
        "English": os.path.join(args.test_base_dir, "english"),
        "Spanish": os.path.join(args.test_base_dir, "spanish")
    }
    
    results = []
    
    # Run evaluation for each language directory
    for language, test_dir in test_dirs.items():
        if not os.path.exists(test_dir):
            print(f"Warning: Test directory {test_dir} does not exist. Skipping {language} evaluation.")
            continue
            
        print(f"\n===== Evaluating {language} =====")
        
        # Step 1: Generate predictions first
        input_file = os.path.join(test_dir, "input.txt")
        pred_file = os.path.join(test_dir, "pred.txt")
        
        predict_cmd = [
            "python", "./src/predict.py",
            "--work_dir", args.work_dir,
            "--test_data", input_file,
            "--test_output", pred_file
        ]
        
        # Step 2: Then evaluate accuracy
        eval_cmd = [
            "python", "./src/eval.py",
            "--test_dir", test_dir
        ]
        
        try:
            # Run prediction
            print(f"Generating predictions for {language}...")
            subprocess.run(predict_cmd, check=True)
            
            # Run evaluation
            print(f"Evaluating accuracy for {language}...")
            output = subprocess.check_output(eval_cmd, universal_newlines=True)
            
            # Extract accuracy using regex
            match = re.search(r"Accuracy is: ([0-9.]+)%", output)
            if match:
                accuracy = float(match.group(1))
                results.append([language, f"{accuracy:.2f}%"])
            else:
                # If regex doesn't match, just show the raw output
                accuracy = output.strip()
                results.append([language, accuracy])
            
            print(output.strip())
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing {language}: {e}")
            results.append([language, "Error"])
    
    # Print summary table
    if results:
        print("\n===== SUMMARY =====")
        print(tabulate(results, headers=["Language", "Accuracy"], tablefmt="grid"))
    else:
        print("\nNo results were generated.")

if __name__ == "__main__":
    main()