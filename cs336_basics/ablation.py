import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from cs336_basics.utils.nn import softmax
from cs336_basics.model import MultiheadSelfAttention, RMSNorm, FFN


class TransformerBlock_ablation(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, attn_pdrop: float, resid_pdrop: float):
        super(TransformerBlock_ablation, self).__init__() 
        # first sub-block
        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, attn_pdrop=attn_pdrop)
        self.drop1 = nn.Dropout(resid_pdrop)
        self.ln1 = RMSNorm(d_model=d_model)

        self.ffn = FFN(d_model=d_model, d_ff=d_ff) 
        self.drop2 = nn.Dropout(resid_pdrop)
        self.ln2 = RMSNorm(d_model=d_model)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.ln1(x + self.drop1(self.attn(x)))
        x = self.ln2(x + self.drop2(self.ffn(x)))
        return x

class TransformerLM_ablation(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            attn_pdrop: float,
            residual_pdrop: float,
        ):
        super(TransformerLM_ablation, self).__init__()
        self.token_embeddings = torch.nn.Embedding(vocab_size, d_model)
        self.position_embeddings = torch.nn.Embedding(context_length, d_model)
        self.drop = torch.nn.Dropout(residual_pdrop)
        self.layers = torch.nn.Sequential(*[TransformerBlock_ablation(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False) 

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        # embedding
        n, seqlen = x.shape
        positions = torch.arange(seqlen, device=x.device).expand(n, seqlen)
        x = self.drop(self.token_embeddings(x) + self.position_embeddings(positions))
        # attn layers
        x = self.layers(x)
        # out layer
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
