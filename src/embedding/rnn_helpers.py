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



def to_sample_and_expected_result(data):
    """
    Splits the provided input into sample, y_true pairs.
    Each line is split at a random index between MIN_INPUT_SIZE and MAX_INPUT_SIZE
    y_true represents the next chracter after the split. The remainder of the string is discarded

    Output:

    x: list of training samples
    y_true: list of expected outputs
    """
    x = []
    y_true = []

    for i, line in enumerate(train_data):
        #The max point we can split at is 1 minus the length of the string since we need one char for y_true
        max_split_index = len(line)-1
        split_index = random.randint(min(MIN_INPUT_SIZE, max_split_index), min(max_split_index, MAX_INPUT_SIZE))
        x.append(line[0:split_index])
        y_true.append(line[split_index])
        

    return x, y_true