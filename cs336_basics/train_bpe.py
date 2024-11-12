import os
from collections import defaultdict, Counter
from typing import Dict, Optional, Tuple, Iterable, List
import regex as re
import json
from cs336_basics.utils.io import PAT

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs):
    # init variables
    print("init vocabs ...")
    vocab_indexes = list(range(33,126+1)) + list(range(161, 172+1)) + list(range(174, 255+1)) + list(range(0,32+1)) + list(range(127,160+1)) + [173]
    vocab = {0: '<|endoftext|>'.encode('utf-8')}
    for i in range(1, len(vocab_indexes)+1):
        vocab[i] = bytes([vocab_indexes[i-1]])
    merges = []

    freq_table = Counter()

    print("processing data line by line...")
    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = re.findall(PAT, line)
            for tok in tokens:
                byte_tuple = tuple(bytes([b]) for b in tok.encode('utf-8'))
                freq_table[byte_tuple] += 1
    print("computing BPE merges ...")
    succesive_pairs = defaultdict(int)
    cur_merge_freq = None 
    for bpe_seq, freq in freq_table.items():
        for i in range(len(bpe_seq)-1):
            succesive_pairs[(bpe_seq[i], bpe_seq[i+1])] += freq
    print("merging BPE ...")
    for vocab_index in range(len(vocab), vocab_size-len(special_tokens)+1):
        if vocab_index % 100 == 0:
            print(vocab_index)
        # compute merge
        merge = max(succesive_pairs, key=lambda x: (succesive_pairs[x], x))
        # merge and add to vocab
        merges.append(merge) # We seem to have space seperators
        vocab[vocab_index] = merge[0]+merge[1]
        # update freq_table
        new_freq_table = defaultdict(int)
        for byte_seq, freq in freq_table.items():
            any_merged = False
            match_1 = False
            new_byte_seq = []
            for byte in byte_seq:
                if (not match_1) and byte == merge[0]:
                    match_1 = True
                    new_byte_seq.append(byte)
                elif match_1 and byte == merge[1]:
                    new_byte_seq.pop()
                    new_byte_seq.append(merge[0]+merge[1])
                    match_1 = False
                    any_merged = True
                elif match_1 and byte == merge[0]:
                    new_byte_seq.append(byte)
                else:
                    new_byte_seq.append(byte)
                    match_1 = False
            if any_merged:
                # remove the counts of the orig byte_seq:
                for i in range(len(byte_seq)-1):
                    succesive_pairs[(byte_seq[i], byte_seq[i+1])] -= freq
                # add the counts of the new byte_seq:
                for i in range(len(new_byte_seq)-1):
                    succesive_pairs[(new_byte_seq[i], new_byte_seq[i+1])] += freq
            new_freq_table[tuple(new_byte_seq)] = freq
        del freq_table
        freq_table = new_freq_table
    for special_token in special_tokens:
        if special_token == '<|endoftext|>':
            continue
        vocab[len(vocab)] = special_token
    return vocab, merges
