import argparse
import time
from tests.adapters import Tokenizer
from tests.adapters import run_train_bpe
import numpy as np

parser = argparse.ArgumentParser(description="Supply data file")
parser.add_argument('--vocab_path', type=str, help="path to input",
                    default="dict_files/vocab_readable.json")
parser.add_argument('--merge_path', type=str, help="path to vocab",
                    default="dict_files/merges_readable.json")
parser.add_argument('--input_path', type=str, help="path to data",
                    default="data/TinyStoriesV2-GPT4-train.txt")
parser.add_argument('--output_path', type=str, help="where to save dataset",
                    default="data/TinyStoriesV2-GPT4-train-encoded.npy")
parser.add_argument('--test', type=str, help="test if it works",
                    default=False)
args = parser.parse_args()
special_tokens=["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_filepath=args.vocab_path, merges_filepath=args.merge_path, special_tokens=special_tokens) 

def encode_data_chunks(input_path, output_path, chunk_size=1024):
    with open(input_path, 'r', encoding='utf-8') as in_file, open(output_path, 'wb') as out_file:
        buffer = []
        for e, line in enumerate(in_file):
            encoded_data = tokenizer.encode(line)
            buffer.extend(encoded_data)
            if (e+1) % chunk_size == 0:
                print("saving chunk: %d" % (e/chunk_size+1))
                output_data = np.array(buffer, dtype=np.uint16)
                output_data.tofile(out_file)
                buffer = []
        if buffer:
            output_data = np.array(buffer, dtype=np.uint16)
            output_data.tofile(out_file)

def test():
    print("testing ...")
    # get text
    def get_data(data_path):
        with open(data_path, 'r') as f:
            contents = f.read()
        return contents
    sample_input_path = "data/TinyStoriesV2-GPT4-sample.txt"
    sample_output_path = "data/TinyStoriesV2-GPT4-sample-encoded.npy"
    print("loading sample data ...")
    data = get_data(sample_input_path)
    print("encoding sample data ...")
    encode_data_chunks(sample_input_path, sample_output_path)
    print("loading encoding sample data back in ...")
    encoded_data = np.fromfile(sample_output_path, dtype=np.uint16)
    encoded_data_list = encoded_data.tolist()
    print("tokenizing it back ...")
    recovered_data = tokenizer.decode(encoded_data_list)
    print("comparing ...")
    print("Is the data fully recovered?", data == recovered_data)
if not args.test:
    encode_data_chunks(args.input_path, args.output_path)
else:
    test()
