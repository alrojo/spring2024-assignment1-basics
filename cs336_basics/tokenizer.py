import os
from collections import defaultdict, Counter
from typing import Dict, Optional, Tuple, Iterable, List
import regex as re
import json
from cs336_basics.utils.io import PAT
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str]=None):
        self.vocab = vocab
        self.special_tokens = special_tokens
        self.merges = merges
        self.inverse_vocab = {v:k for k, v in self.vocab.items()} if self.vocab is not None else None
        self.merges_dict = {v:k for k,v in enumerate(self.merges)}
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)
        with open(merges_filepath, 'r', encoding='utf-8') as merges_file:
            merges = json.load(merges_file)
        vocab = {int(k): v.encode('ISO-8859-1') if isinstance(v, str) else v for k, v in vocab.items()}
        merges = [(pair[0].encode('ISO-8859-1'), pair[1].encode('ISO-8859-1')) for pair in merges]
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def tokenize_special_tokens(self, text):
        escaped_tokens = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
        pattern = '|'.join(escaped_tokens)
        tokens = re.split(f'({pattern})', text)
        tokens = [token for token in tokens if token]
        return tokens

    def encode(self, text: str):
        if self.special_tokens is not None:
            outs = []
            text_parts = self.tokenize_special_tokens(text)
            for e, text_part in enumerate(text_parts):
                if text_part in self.special_tokens:
                    outs.append([self.inverse_vocab[bytes(text_part.encode('utf-8'))]])
                else:
                    outs.append(self.encode_bits(text_part))
            return [item for sublist in outs for item in sublist]
        else:
            return self.encode_bits(text)


    def encode_bits(self, text: str):
        # pre tok text
        tokens = re.findall(PAT, text)
        byte_tokens = []
        def contains_unicode(s):
            return any(ord(char) > 127 for char in s)
        for tok in tokens:
            byte_tuple = []
            #for char in tok:
            #    byte_tuple.append(char.encode('utf-8'))
            #byte_tuple = tuple(byte_tuple)
            int_tuple = tok.encode("utf-8")
            byte_tuple = [bytes([a]) for a in int_tuple]
            byte_tokens.append(byte_tuple)

        # Now walk through merges in order for each sub-token
        encoded_tokens = []
        for byte_seq in byte_tokens:
            byte_seq = list(byte_seq)
            # find lowest merge
            while True:
                pairs = [(a,b) for a,b in zip(byte_seq[:-1], byte_seq[1:])]
                if len(pairs)==0:
                    break
                merge_ids = []
                for pair in pairs:
                    merge_ids.append(self.merges_dict.get(pair, len(self.merges)))
                min_merge_value = min(merge_ids)
                if min_merge_value == len(self.merges):
                    break
                min_merge_index = merge_ids.index(min_merge_value)
                byte_seq[min_merge_index] = byte_seq[min_merge_index] + byte_seq[min_merge_index+1]
                byte_seq.pop(min_merge_index+1)
            encoded_bpe = []
            for bpe in byte_seq:
                encoded_bpe.append(self.inverse_vocab[bpe])
            encoded_tokens += encoded_bpe
        return encoded_tokens

    def encode_iterable(self, iterable: Iterable[str]):
        for item in iterable:
            for encoded_token in self.encode(item):
                yield encoded_token

    def decode(self, ids: List[int]):
        bpes = []
        for elem in ids:
            bpe = self.vocab[elem]
            bpes.append(bpe)
        bpes = b''.join(bpes)
        text = bpes.decode('utf-8', errors='replace')
        return text

def get_tokenizer(
    vocab: Dict[int, bytes],
    merges: List[tuple[bytes, bytes]],
    special_tokens: Optional[List[str]] = None,
):
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab: dict[int, bytes]
            The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges: list[tuple[bytes, bytes]]
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens: Optional[list[str]]
            A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)
