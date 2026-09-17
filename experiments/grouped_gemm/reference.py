"""Independent dequantization and grouped FP32 products for MXFP8 inputs."""

import torch


def _unpack_scales(flat, rows, sf_blocks):
    words_per_row = (sf_blocks + 3) // 4
    packed = flat.reshape(words_per_row, rows).T.contiguous().to(torch.int64)
    unpacked = torch.empty((rows, words_per_row * 4), device=flat.device, dtype=torch.uint8)
    for byte in range(4):
        unpacked[:, byte::4] = ((packed >> (8 * byte)) & 0xFF).to(torch.uint8)
    return torch.pow(2.0, unpacked[:, :sf_blocks].float() - 127.0)


def reference(a, b, sfa_flat, sfb_flat, offsets, storage_offsets, transpose_b=False, sf_granularity_k=128):
    m_storage, k = a.shape
    m_total = int(offsets[-1].item())
    experts = b.shape[0]
    n = b.shape[1] if transpose_b else b.shape[2]
    sf_blocks = (k + sf_granularity_k - 1) // sf_granularity_k
    sfa = _unpack_scales(sfa_flat, m_storage, sf_blocks)
    sfb = _unpack_scales(sfb_flat, experts * n, sf_blocks).view(experts, n, sf_blocks)
    outputs = torch.empty((m_total, n), device=a.device, dtype=torch.float32)

    for group in range(experts):
        start, end = int(offsets[group].item()), int(offsets[group + 1].item())
        input_start = int(storage_offsets[group].item())
        input_end = input_start + end - start
        out = torch.zeros((end - start, n), device=a.device, dtype=torch.float32)
        for sf_block in range(sf_blocks):
            k_start = sf_block * sf_granularity_k
            k_end = min(k_start + sf_granularity_k, k)
            lhs = a[input_start:input_end, k_start:k_end].float() * sfa[input_start:input_end, sf_block : sf_block + 1]
            weight = b[group].float()
            scale = sfb[group, :, sf_block : sf_block + 1]
            if transpose_b:
                rhs = weight[:, k_start:k_end] * scale
                out += lhs @ rhs.T
            else:
                rhs = weight[k_start:k_end, :] * scale.T
                out += lhs @ rhs
        outputs[start:end] = out

    return outputs.to(torch.bfloat16)
