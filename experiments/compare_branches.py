"""Compare experiment contracts across checkouts using only the standard library.

Example: python -m experiments.compare_branches . ../tilelang-dev-a100 ../tilelang-dev-b200
The check includes case semantics, ordered config IDs, active Carver adapters,
and kernel sources. Backend selection and GEMM/attention builders may differ.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROBE = """
import ast, hashlib, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from experiments.families import FAMILIES, family_module
from experiments.suite import core_cases
from experiments.backend import NAME, FP8_COMPUTE_DTYPE
root = pathlib.Path(sys.argv[1])
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def source(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
operations = {}
for op, family in FAMILIES.items():
    cases = family_module(op, 'cases')
    pool = family_module(op, 'spaces').get_configs()
    operations[op] = dict(family=family, development=[w.to_dict() for w in cases.cases()],
        final=[w.to_dict() for w in cases.cases(holdout=True)],
        training_validation=[w.to_dict() for w in cases.training_cases()],
        configs=len(pool), pool_sha256=digest(pool),
        kernel_sha256=source(root/f'experiments/{family}/kernel.py'),
        carver_sha256=source(root/f'experiments/{family}/carver.py'))
shared = ['tilelang/carver/matmul_analysis.py', 'experiments/common/carver.py', 'experiments/gemm/carver.py',
          'tilelang/carver/template/matmul_fp8.py',
          'examples/gemm_fp8/example_tilelang_gemm_fp8.py',
          'examples/kda/chunk_intra_token_parallel.py',
          'experiments/gemm_fp8/reference.py', 'experiments/grouped_gemm/reference.py',
          'experiments/kda/reference.py', 'examples/grouped_gemm/example_grouped_gemm_fwd.py']
shared += [str(p.relative_to(root)) for folder in ('arch', 'roller')
           for p in sorted((root/'tilelang/carver'/folder).rglob('*.py'))]
shared += ['tilelang/carver/template/' + name for name in
           ('matmul.py', 'flashattention.py', 'kda_chunk.py', 'graph.py', 'base.py')]
assert len(core_cases('final')) == sum(len(v['final']) for v in operations.values())
assert not any(x.split('.')[0] in ('torch','tilelang','tvm','xgboost') for x in sys.modules)
shared_sources = {p:source(root/p) for p in shared}
grouped = ast.parse((root/'tilelang/carver/template/grouped_matmul.py').read_text())
active = next(n for n in grouped.body if isinstance(n, ast.ClassDef) and n.name == 'GroupedMatmulTemplate')
shared_sources['tilelang/carver/template/grouped_matmul.py:GroupedMatmulTemplate'] = digest(ast.dump(active))
print(json.dumps(dict(backend=NAME, fp8_compute_dtype=FP8_COMPUTE_DTYPE, operations=operations,
                     shared_sources=shared_sources)))
"""


def compare(roots):
    checkouts = []
    for root in roots:
        root = Path(root).resolve()
        probe = json.loads(subprocess.check_output([sys.executable, "-I", "-S", "-c", PROBE, str(root)], text=True))
        probe["root"] = str(root)
        probe["branch"] = subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"], text=True).strip()
        checkouts.append(probe)
    reference, differences, allowed = checkouts[0], [], []
    for other in checkouts[1:]:
        if set(reference["operations"]) != set(other["operations"]):
            differences.append(dict(branch=other["branch"], field="operations"))
        for op in reference["operations"].keys() & other["operations"].keys():
            for field, value in reference["operations"][op].items():
                if value == other["operations"][op][field]:
                    continue
                item = dict(branch=other["branch"], operation=op, field=field)
                (allowed if field == "kernel_sha256" and op in ("gemm", "attention") else differences).append(item)
        for path, value in reference["shared_sources"].items():
            if value != other["shared_sources"][path]:
                differences.append(dict(branch=other["branch"], source=path))
        if reference["fp8_compute_dtype"] != other["fp8_compute_dtype"]:
            allowed.append(dict(branch=other["branch"], field="fp8_compute_dtype", value=other["fp8_compute_dtype"]))
    return dict(compatible=not differences, differences=differences, allowed_backend_differences=allowed, checkouts=checkouts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkouts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if len(args.checkouts) < 2:
        parser.error("supply at least two checkouts")
    report = compare(args.checkouts)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checkouts"}, indent=2))
    for checkout in report["checkouts"]:
        pools = ", ".join(f"{op}={v['configs']}" for op, v in checkout["operations"].items())
        print(f"{checkout['branch']}: {pools}")
    return int(not report["compatible"])


if __name__ == "__main__":
    raise SystemExit(main())
