"""Carver ranks Ampere's BF16-compute emulation over the exact FP8 pool."""


def carver_rank(workload, device, configs, top_k):
    from experiments.common.carver import _architecture
    from experiments.gemm.carver import rank_configs
    from tilelang.carver.template import MatmulTemplate

    p = workload.parameters
    template = MatmulTemplate(
        M=p["m"],
        N=p["n"],
        K=p["k"],
        trans_A=False,
        trans_B=True,
        in_dtype="bfloat16",
        out_dtype="bfloat16",
        accum_dtype="float32",
        _arch=_architecture(device.target),
    )
    report = rank_configs(
        configs,
        m=p["m"],
        n=p["n"],
        k=p["k"],
        dtype="bfloat16",
        target=device.target,
        top_k=top_k,
        transpose_b=True,
        template=template,
        thread_key="threads",
    )
    report.update(
        metric="carver_traffic_waves",
        score_units="byte-waves",
        implementation="e4m3_storage_bf16_tensorcore_emulation",
    )
    report["assumptions"].append(
        "Ampere converts E4M3 storage to BF16 tiles; Carver models the BF16 tensor-core compute and conservatively charges BF16 operand traffic."
    )
    return report
