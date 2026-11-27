import torch
import math

def kv_cache_decode_step(
    query: torch.Tensor,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
) -> dict:

    # 1. Append new K and V to the existing cache
    new_key_cache = torch.cat(
        [key_cache, new_key.unsqueeze(2)], dim=2
    )

    new_value_cache = torch.cat(
        [value_cache, new_value.unsqueeze(2)], dim=2
    )

    # 2. GQA group size
    group_size = num_query_heads // num_kv_heads

    # 3. Perform attention in float32
    query_fp32 = query.float()
    key_fp32 = new_key_cache.float()
    value_fp32 = new_value_cache.float()

    head_dim = query.shape[-1]

    outputs = []

    # 4. Compute attention for each query head
    for q_head in range(num_query_heads):

        # Map query head to its corresponding KV head
        kv_head_index = q_head // group_size

        # Extract query, keys and values
        q = query_fp32[:, q_head, :]
        k = key_fp32[:, kv_head_index, :, :]
        v = value_fp32[:, kv_head_index, :, :]

        # 5. Compute attention scores
        scores = torch.matmul(
            q.unsqueeze(1),
            k.transpose(-2, -1)
        )

        # 6. Scale scores
        scores = scores / math.sqrt(head_dim)

        # 7. Softmax over sequence positions
        attention_weights = torch.softmax(scores, dim=-1)

        # 8. Weighted sum of values
        head_output = torch.matmul(
            attention_weights,
            v
        ).squeeze(1)

        outputs.append(head_output)

    # 9. Combine all query heads
    output = torch.stack(outputs, dim=1)

    # 10. Restore original dtype
    output = output.to(query.dtype)

    # 11. Return outputs and updated caches
    return {
        "output": output,
        "new_key_cache": new_key_cache,
        "new_value_cache": new_value_cache
    }
