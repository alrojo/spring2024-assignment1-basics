import argparse
import time
from tests.adapters import Tokenizer
from tests.adapters import run_train_bpe

parser = argparse.ArgumentParser(description="Supply data file")
parser.add_argument('--vocab_path', type=str, help="path to input",
                    default="dict_files/vocab_readable.json")
parser.add_argument('--merge_path', type=str, help="path to vocab",
                    default="dict_files/merges_readable.json")
args = parser.parse_args()
special_tokens=["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_filepath=args.vocab_path, merges_filepath=args.merge_path, special_tokens=special_tokens) 
longest_BPE = ""
for key, value in tokenizer.vocab.items():
    BPE = value.decode("utf-8", errors="replace")
    if len(BPE) > len(longest_BPE):
        print("updating: ", len(BPE), BPE)
        longest_BPE = BPE
