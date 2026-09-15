"""The CUDA adapter exports facts that the installed core can replay alone."""

import json
from pathlib import Path
import subprocess
import sys

from tilelang.tiletune import analyze_prim_func, TileTuneConfig
from test_analysis import gemm
from test_cost import LIMITS


def test_cuda_facts_replay_without_a_compiler(tmp_path):
    path = tmp_path / "facts.json"
    result = analyze_prim_func(
        gemm(),
        TileTuneConfig(ranking_metric="traffic_waves", facts_path=str(path)),
        target={"kind": "cuda", "arch": "sm_80"},
        device_limits=LIMITS,
    )
    from tiletune_core import KernelFacts
    from tiletune_core.cuda import evaluate_cuda_facts

    facts = KernelFacts.from_dict(json.loads(path.read_text()))
    assert evaluate_cuda_facts(facts).score == result["tile_cost"]["score"]
    root = str(Path(__file__).resolve().parents[3])
    code = f"""import sys, json
sys.path.insert(0, {root!r})
from tiletune_core import KernelFacts
from tiletune_core.cuda import evaluate_cuda_facts
r = evaluate_cuda_facts(KernelFacts.from_dict(json.load(open({str(path)!r}))))
assert r.score == {result["tile_cost"]["score"]!r}
assert not any(k.split('.')[0] in ('tilelang', 'torch', 'tvm') for k in sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", code], check=True)
