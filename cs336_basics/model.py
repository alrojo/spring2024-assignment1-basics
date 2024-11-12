import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from cs336_basics.utils.nn import softmax

def scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    seq_len, d_k = K.shape[-2:]
    K_transposed = K.transpose(-2,-1)
    QKT = torch.matmul(Q, K_transposed)
    QKT_norm = QKT/d_k**0.5 
    if mask is not None:
        QKT_norm.masked_fill_(mask, float('-inf'))
    attn_vals = softmax(QKT_norm, dim=-1) 
    if pdrop is not None:
        attn_vals = nn.functional.dropout(attn_vals, pdrop)
    out = torch.matmul(attn_vals, V)
    return out

class GELU(nn.Module):
    def forward(self, x :torch.FloatTensor) -> torch.FloatTensor:
        return x*0.5*(1+torch.erf(x/np.sqrt(2)))

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean_norm = torch.mean(x ** 2, dim=-1, keepdim=True)
        rms_norm = torch.sqrt(mean_norm + self.eps)
        norm = self.weight / rms_norm
        out_features = x * norm
        return out_features

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.gelu = GELU()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.gelu(self.w1(x))
        x = self.w2(x)
        return x

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float):
        super(MultiheadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.d_k = d_model // num_heads
        #self.qkv_proj = nn.Linear(d_model, d_model*3, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        n, seqlen, d_model = x.shape
        #qkv = self.qkv_proj(x)
        #q = qkv[:,:,0]
        #qkv = qkv.view(n, seqlen, 3, self.num_heads, self.d_k)
        #q, k, v = qkv[:,:,0].transpose(1,2), qkv[:,:,1].transpose(1,2), qkv[:,:,2].transpose(1,2)

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(n, seqlen, self.num_heads, self.d_k).transpose(1,2)
        k = k.view(n, seqlen, self.num_heads, self.d_k).transpose(1,2)
        v = v.view(n, seqlen, self.num_heads, self.d_k).transpose(1,2)
        mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=1).bool().to(x.device)
        x = scaled_dot_product_attention(k, q, v, mask=mask, pdrop=self.attn_pdrop)
        x = x.transpose(1, 2)
        x = x.contiguous().view(n, seqlen, self.d_model)
        x = x.view(n, seqlen, self.d_model)
        x = self.output_proj(x)
        return x

    def load_state_dict_custom(self, state_dict):
        weights = state_dict
        for i in range(self.num_heads):
            self.q_proj.weight.data[i*self.d_k:(i+1)*self.d_k] = weights[f"q_heads.{i}.weight"]
            self.k_proj.weight.data[i*self.d_k:(i+1)*self.d_k] = weights[f"k_heads.{i}.weight"]
            self.v_proj.weight.data[i*self.d_k:(i+1)*self.d_k] = weights[f"v_heads.{i}.weight"]
        self.output_proj.weight.data = weights['output_proj.weight']
        """
    def load_state_dict_custom(self, state_dict):
        weights = state_dict
        Q_start, K_start, V_start = 0, self.d_model, 2*self.d_model
        for i in range(self.num_heads):
            head_index = self.d_k*3*i
            Q_idx, K_idx, V_idx = Q_start + i*self.d_k, K_start + i*self.d_k, V_start + i*self.d_k
            self.qkv_proj.weight.data[Q_idx:Q_idx+self.d_k] = weights[f"q_heads.{i}.weight"]
            self.qkv_proj.weight.data[K_idx:K_idx+self.d_k] = weights[f"k_heads.{i}.weight"]
            self.qkv_proj.weight.data[V_idx:V_idx+self.d_k] = weights[f"v_heads.{i}.weight"]
        self.out_proj.weight.data = weights['output_proj.weight']

        d_key, d_value = weights["q_heads.0.weight"].shape[0], weights["v_heads.0.weight"].shape[0]
        d_n = 2*d_key + d_value
        sdp = []
        for i in range(num_heads):
            Qi = torch.matmul(in_features, weights[f"q_heads.{i}.weight"].T)
            Ki = torch.matmul(in_features, weights[f"k_heads.{i}.weight"].T)
            Vi = torch.matmul(in_features, weights[f"v_heads.{i}.weight"].T)
            sdpi = run_scaled_dot_product_attention(Q=Qi, K=Ki, V=Vi, mask=None, pdrop=None)
            torch_version = torch.nn.functional.scaled_dot_product_attention(Qi, Ki, Vi)
            print(sdpi)
            print(torch_version)
            print("hello?")
            assert False
            sdp.append(sdpi)
        sdp = torch.cat(sdp, dim=-1)
        out = torch.matmul(sdp, weights["output_proj.weight"].T)
        return out
        concat_weights = []
        for i in range(num_heads):
            concat_weights.append(weights[f"q_heads.{i}.weight"])
            concat_weights.append(weights[f"k_heads.{i}.weight"])
            concat_weights.append(weights[f"v_heads.{i}.weight"])
        concat_weights = torch.cat(concat_weights, dim=0)
        # Matmul 1: compute QKV for each head
        QKVn = torch.matmul(in_features, concat_weights.T) 
        Qn, Kn, Vn = [], [], []
        # slice up and partition into Q, K, V
        for i in range(num_heads):
            head_index = d_n*i
            Q_start, Q_end = head_index, head_index + d_key
            K_start, K_end = head_index + d_key, head_index + 2*d_key
            V_start, V_end = head_index + 2*d_key, head_index + d_n
            Qn.append(QKVn[..., Q_start:Q_end].unsqueeze(1))
            Kn.append(QKVn[..., K_start:K_end].unsqueeze(1))
            Vn.append(QKVn[..., V_start:V_end].unsqueeze(1))
        Qn = torch.cat(Qn, dim=1)
        Kn = torch.cat(Kn, dim=1)
        Vn = torch.cat(Vn, dim=1)
        # Matmul 2: attn
        sdp = run_scaled_dot_product_attention(Q=Qn, K=Kn, V=Vn, mask=None, pdrop=attn_pdrop)
        # concat heads
        sdp = sdp.permute(0, 2, 3, 1)
        sdp = torch.reshape(sdp, (sdp.shape[0], sdp.shape[1], -1))
        # Matmul 2: output 
        out = torch.matmul(sdp, weights["output_proj.weight"].T)
        return out 
"""
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, attn_pdrop: float, resid_pdrop: float):
        super(TransformerBlock, self).__init__() 
        # first sub-block
        self.ln1 = RMSNorm(d_model=d_model)
        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, attn_pdrop=attn_pdrop)
        self.drop1 = nn.Dropout(resid_pdrop)

        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff) 
        self.drop2 = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x

class TransformerLM(nn.Module):
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
        super(TransformerLM, self).__init__()
        self.token_embeddings = torch.nn.Embedding(vocab_size, d_model)
        self.position_embeddings = torch.nn.Embedding(context_length, d_model)
        self.drop = torch.nn.Dropout(residual_pdrop)
        self.layers = torch.nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
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
