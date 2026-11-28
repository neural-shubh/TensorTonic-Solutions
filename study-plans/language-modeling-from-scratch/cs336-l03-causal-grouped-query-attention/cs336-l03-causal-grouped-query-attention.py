import torch
import math

def causal_gqa(q, k, v):
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    seq_len = q.shape[-2]
    head_dim = q.shape[-1]

    group_size = num_q_heads // num_kv_heads

    # Expand K and V heads to match Q heads
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)

    # Attention scores
    score = torch.matmul(q, k.transpose(-2, -1))
    score = score / math.sqrt(head_dim)

    # Causal mask
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
        diagonal=1
    )

    score = score.masked_fill(mask, float("-inf"))

    # Softmax
    attention_weights = torch.softmax(score, dim=-1)

    # Attention output
    output = torch.matmul(attention_weights, v)

    return output