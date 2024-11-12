import argparse, os, time
from tests.adapters import Tokenizer
from tests.adapters import run_train_bpe
import numpy as np

parser = argparse.ArgumentParser(description="Supply data file")
parser.add_argument('--save_folder', type=str, help="path to vocab folder",
                    default="experiments/data/TinyStoriesV2-GPT4-sample-500")
parser.add_argument('--input_path', type=str, help="path to data",
                    default="experiments/downloads/TinyStoriesV2-GPT4-sample.txt")
parser.add_argument('--test', type=str, help="test if it works",
                    default=False)
args = parser.parse_args()

vocab_path = os.path.join(args.save_folder, "vocab_readable.json")
merges_path = os.path.join(args.save_folder, "merges_readable.json")


print("initializing vocab from %s ..." % args.save_folder)
special_tokens=["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=special_tokens) 

def encode_data_chunks(input_path, output_path, chunk_size=1024):
    total_tokens = 0
    with open(input_path, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            total_tokens += len(tokenizer.encode(line))
    arr = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(total_tokens,))
    with open(input_path, 'r', encoding='utf-8') as in_file:
        idx=0
        buffer = []
        for e, line in enumerate(in_file):
            encoded_data = tokenizer.encode(line)
            buffer.extend(encoded_data)
            if (e+1) % chunk_size == 0:
                arr[idx:idx+len(buffer)] = buffer
                print("saving chunk: %d" % (e // chunk_size+1))
                idx += len(buffer)
                buffer = []
        if buffer:
            arr[idx:idx+len(buffer)] = buffer
    arr.flush()

dataset_name = "%s-encoded.bin" % (os.path.splitext(os.path.basename(args.input_path))[0])
output_path = os.path.join(args.save_folder, dataset_name)
encode_data_chunks(args.input_path, output_path)
print("saved to %s" % output_path)

if args.test:
    print("testing that text reverses ...")
    # get text
    def get_data(data_path):
        with open(data_path, 'r') as f:
            contents = f.read()
        return contents
    print("loading raw text data ...")
    data = get_data(args.input_path)
    print("loading encoding sample data back in ...")
    encoded_data = np.memmap(output_path, dtype=np.uint16, mode='r')
    encoded_data_list = encoded_data.tolist()
    print("tokenizing it back ...")
    recovered_data = tokenizer.decode(encoded_data_list)
    print("comparing ...")
    print("Is the data fully recovered?", data == recovered_data)
