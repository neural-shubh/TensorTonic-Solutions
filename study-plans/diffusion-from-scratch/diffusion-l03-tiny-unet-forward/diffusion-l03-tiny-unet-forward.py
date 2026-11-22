
import torch
import torch.nn.functional as f


def residual_block(x, time_emb, params, prefix, num_groups):

    # First normalization and convolution
    h = f.group_norm(
        x,
        num_groups=num_groups,
        weight=params[f"{prefix}_norm1_weight"],
        bias=params[f"{prefix}_norm1_bias"],
        eps=1e-5
    )

    h = f.silu(h)

    h = f.conv2d(
        h,
        params[f"{prefix}_conv1_weight"],
        params[f"{prefix}_conv1_bias"],
        padding=1
    )

    # Project timestep embedding
    t = f.linear(
        f.silu(time_emb),
        params[f"{prefix}_time_weight"],
        params[f"{prefix}_time_bias"]
    )

    h = h + t[:, :, None, None]

    # Second normalization and convolution
    h = f.group_norm(
        h,
        num_groups=num_groups,
        weight=params[f"{prefix}_norm2_weight"],
        bias=params[f"{prefix}_norm2_bias"],
        eps=1e-5
    )

    h = f.silu(h)

    h = f.conv2d(
        h,
        params[f"{prefix}_conv2_weight"],
        params[f"{prefix}_conv2_bias"],
        padding=1
    )

    # Residual shortcut
    if f"{prefix}_skip_weight" in params:
        residual = f.conv2d(
            x,
            params[f"{prefix}_skip_weight"],
            params[f"{prefix}_skip_bias"]
        )
    else:
        residual = x

    return h + residual


def tiny_unet_forward(x, timesteps, params, num_groups):

    # 1. Sinusoidal timestep embedding
    E = params["time_linear1_weight"].shape[1]

    half_dim = E // 2

    indices = torch.arange(
        half_dim,
        device=x.device,
        dtype=x.dtype
    )

    frequencies = 10000 ** (-2 * indices / E)

    angles = (
        timesteps.to(x.dtype)[:, None]
        * frequencies[None, :]
    )

    time_emb = torch.cat(
        [torch.cos(angles), torch.sin(angles)],
        dim=1
    )

    # 2. Time MLP
    time_emb = f.linear(
        time_emb,
        params["time_linear1_weight"],
        params["time_linear1_bias"]
    )

    time_emb = f.silu(time_emb)

    time_emb = f.linear(
        time_emb,
        params["time_linear2_weight"],
        params["time_linear2_bias"]
    )

    # 3. Input convolution
    h = f.conv2d(
        x,
        params["conv_in_weight"],
        params["conv_in_bias"],
        padding=1
    )

    # 4. Encoder residual block
    h1 = residual_block(
        h, time_emb, params, "down_res", num_groups
    )

    # 5. Downsampling
    h = f.conv2d(
        h1,
        params["downsample_weight"],
        params["downsample_bias"],
        stride=2,
        padding=1
    )

    # 6. Middle residual block
    h = residual_block(
        h, time_emb, params, "mid_res", num_groups
    )

    # 7. Upsampling
    h = f.interpolate(
        h,
        scale_factor=2,
        mode="nearest"
    )

    h = f.conv2d(
        h,
        params["upsample_weight"],
        params["upsample_bias"],
        padding=1
    )

    # 8. Skip connection
    h = torch.cat([h, h1], dim=1)

    # 9. Decoder residual block
    h = residual_block(
        h, time_emb, params, "up_res", num_groups
    )

    # 10. Output normalization
    h = f.group_norm(
        h,
        num_groups=num_groups,
        weight=params["out_norm_weight"],
        bias=params["out_norm_bias"],
        eps=1e-5
    )

    h = f.silu(h)

    # 11. Output convolution
    output = f.conv2d(
        h,
        params["conv_out_weight"],
        params["conv_out_bias"],
        padding=1
    )

    return output
