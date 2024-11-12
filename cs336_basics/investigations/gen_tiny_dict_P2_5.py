#!/usr/bin/env python3
import json
import time
import argparse
import os

from tests.adapters import run_train_bpe

# Expected use testing
# python scripts/gen_tiny_dict.py --input_path data/TinyStoriesV2-GPT4-sample.txt --out_name sample --vocab_size 500
# Expected use for the Tiny dataset 
# python scripts/gen_tiny_dict.py --input_path data/TinyStoriesV2-GPT4-sample.txt --out_name TinyStories_10k --vocab_size 10000
parser = argparse.ArgumentParser(description="Supply data file")
parser.add_argument('--input_path', type=str, required=True, help="path to input")
parser.add_argument('--out_name', type=str, required=True, help="path to vocab")
parser.add_argument('--vocab_size', type=int, required=True, help="size of vocabulary")
args = parser.parse_args()

start_time = time.time()
vocab, merges = run_train_bpe(
    input_path=args.input_path,
    vocab_size=args.vocab_size,
    special_tokens=["<|endoftext|>"],
)
end_time = time.time()
print("Time (mins):", (end_time-start_time)/60)

def serialize_vocab_and_merges(vocab, merges, vocab_path, merges_path):
    vocab_serializable = {k: v.decode('ISO-8859-1') if isinstance(v, bytes) else v for k, v in vocab.items()}
    merges_serializable = [(pair[0].decode('ISO-8859-1'), pair[1].decode('ISO-8859-1')) for pair in merges]
    with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab_serializable, vocab_file, ensure_ascii=False, indent=4)
    with open(merges_path, 'w', encoding='utf-8') as merges_file:
        json.dump(merges_serializable, merges_file, ensure_ascii=False, indent=4)

vocab_path = 'dict_files/%s_vocab_readable.json' % args.out_name
merges_path = 'dict_files/%s_merges_readable.json' % args.out_name

serialize_vocab_and_merges(vocab, merges, vocab_path, merges_path)
