import os

MIN_INPUT_SIZE = 3
MAX_INPUT_SIZE = 300

# TODO: For neural network approch later. Need to get fully working
def to_embeddings(line: torch.tensor, vocab_idx):
    # Convert each codepoint in the input into a one hot vector based on the provided vocab map
    output = torch.zeros(len(line), len(vocab_idx))
    for i,c in enumerate(line):
        index = vocab_idx[c]
        print(f"The index for codepoint: {c} is {index}")
        output[index] = 1

    return output

def all_to_embeddings(input: torch.tensor, vocab_idx):
    output = torch.tensor((len(input), MAX_INPUT_SIZE, len(vocab_idx)))

    for i, line in enumerate(input):
        output[i] = to_embeddings(line, vocab_idx)

    return output

def to_codepoint_tensor(data, max_line_size):
    """
    Converts the input into a tensor of codepoints of max_line_size. If the inputs are shorter they padded.
    It is not valid for any line in the input to this function to be over max_line_size

    """
    res = torch.zeros(len(train_data),max_line_size )

    for i, line in enumerate(train_data):
        codepoints = torch.tensor([ord(c) for c in line], dtype=torch.int32)
        res[i] = torch.nn.functional.pad(codepoints, (0, MAX_INPUT_SIZE - len(codepoints)), value=0)

    return res



