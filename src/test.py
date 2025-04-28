import mwxml
import sentencepiece as spm

# dump = mwxml.Dump.from_file(open("/mnt/d/wikidump/wiki_20241201_simple.xml/wiki_20241201_simple.xml"))
# print(dump.site_info.name, dump.site_info.dbname)

# # loop through pages and remove [[]]
# with open("output.txt", "w") as output_file:
#     for page in dump.pages:
#         for revision in page:
#             if revision.text is None:
#                 continue
#             out_str = revision.text.replace("[[", "").replace("]]", "")
#             # and save the updated text to a new file
#             output_file.write(out_str)
#             output_file.write("\n")

spm.SentencePieceTrainer.Train(
    '--input=output.txt --model_prefix=spm --vocab_size=10000 --character_coverage=0.99 --model_type=bpe'
)

# Load trained BPE tokenizer, this is moved to dataset class
# sp = spm.SentencePieceProcessor()
# sp.load("spm.model")

# seq_length = 30  #TODO: Length of input sequences

# # Original text data
# with open("output.txt") as f:
#     with open("inter.csv", "w") as f_csv:
#         # print how many lines are in the file
#         print("Number of lines in the file:", sum(1 for line in f))
#         f.seek(0)
#         current_line = 0
#         # write the header
#         f_csv.write("input\ttarget\n")
#         for line in f:
#             # print every 100000 lines
#             current_line += 1
#             if current_line % 100000 == 0:
#                 print(f"Processing line {current_line}")
#             data = line[:-1]  # the last character is a newline
#             # Encode the text into BPE tokens
#             bpe_tokens = sp.encode(data, out_type=int)  # Encoded as integers
#             for i in range(len(bpe_tokens) - seq_length):
#                 f_csv.write(f"{bpe_tokens[i:i+seq_length]}\t{bpe_tokens[i+seq_length]}\n")
#         print(f"Processed {current_line} lines")
