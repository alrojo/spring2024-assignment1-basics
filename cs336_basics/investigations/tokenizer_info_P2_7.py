import argparse
import time
from tests.adapters import Tokenizer
from tests.adapters import run_train_bpe

parser = argparse.ArgumentParser(description="Supply data file")
parser.add_argument('--vocab_path', type=str, help="path to input",
                    default="dict_files/vocab_readable.json")
parser.add_argument('--merge_path', type=str, help="path to vocab",
                    default="dict_files/merges_readable.json")
parser.add_argument('--data_path', type=str, help="path to data",
                    default="data/TinyStoriesV2-GPT4-sample.txt")
args = parser.parse_args()
special_tokens=["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_filepath=args.vocab_path, merges_filepath=args.merge_path, special_tokens=special_tokens) 

# sample documents

def get_data(data_path):
    with open(data_path, 'r') as f:
        contents = f.read()
    return contents
data = get_data(args.data_path)
start_time = time.time()
data_encoded = tokenizer.encode(data)
end_time = time.time()
time_used_seconds = end_time - start_time

num_bytes = len(bytes(data.encode('utf-8')))
num_tokens = len(data_encoded)

bytes_tokens = num_bytes / num_tokens
bytes_seconds = num_bytes / time_used_seconds
pile_size = 825e9# 825GB as bytes
time_for_pile_hours = pile_size / bytes_seconds / 3600
print("a) For TinyStories we achieve: %.2f bytes / tokens" % bytes_tokens)
print("b) Don't have the compute")
print("c i.) bytes / seconds: %.2f" % bytes_seconds)
print("c ii.) Encoding Pile would take: %.2f hours or %.2f days" % (time_for_pile_hours, time_for_pile_hours/24))
