# H200 results

The current experiment uses the 512-config intra-chunk pool (space version 8).
The saved `kda_chunk_*` JSON files describe historical chunk-output kernels;
they cannot serve as intra-chunk oracles. A complete intra-chunk sweep and
winner validation have not yet been recorded.

Earlier JSON records were moved to
`experiments/results/pre-three-single-pools-20260916/kda/heuristics/H200/`.
They retain the source hashes and timings of their original kernels and pools;
they are not optima for the current experiment. New results should record their
own pool, compilation/correctness outcomes, contention checks and provenance.
