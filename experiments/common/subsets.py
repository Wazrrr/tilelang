"""Timing-independent greedy pairwise coverage with stable configuration IDs."""

import heapq
from itertools import combinations
import json

from .spaces import canonical_config, config_id, legality_reason

SUBSET_VERSION = 1


def pairwise_subset(workload, device, configs, limit):
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("subset budget must be a positive integer or None")
    unique, aliases, rejected = {}, [], []
    for index, config in enumerate(configs):
        # Explicit accelerator-worker grids use their own native knobs.
        reason = legality_reason(workload, device, config) if device.target["kind"] in ("cuda", "hip") else None
        if reason:
            rejected.append(dict(index=index, reason=reason))
            continue
        canonical = canonical_config(workload, config, device)
        key = config_id(canonical)
        if key in unique:
            aliases.append(dict(index=index, representative=unique[key][0]))
        else:
            unique[key] = (index, canonical)
    knobs = sorted({key for _, config in unique.values() for key in config})
    features = {}
    for identity, (_, config) in unique.items():
        values = [(key, json.dumps(config[key], sort_keys=True)) if key in config else (key, "<absent>") for key in knobs]
        features[identity] = set(combinations(values, 2)) | {(value,) for value in values}
    total = set().union(*features.values()) if features else set()
    covered, selected = set(), []
    # Lazy greedy set cover: cached gains are upper bounds after each choice.
    # The canonical hash breaks all ties, including zero-gain remaining slots.
    heap = [(-len(features[key]), key) for key in unique]
    if limit is None or limit >= len(unique):
        # A full oracle includes every meaningful candidate. Greedy ordering
        # cannot change membership, so avoid repeated set-cover work.
        selected = [index for index, _ in unique.values()]
        covered = total
        heap = []
    heapq.heapify(heap)
    while heap and (limit is None or len(selected) < limit):
        _, key = heapq.heappop(heap)
        gain = len(features[key] - covered)
        if heap and (-gain, key) > heap[0]:
            heapq.heappush(heap, (-gain, key))
            continue
        selected.append(unique[key][0])
        covered.update(features[key])
    # Evaluation order retains original grid order and tie semantics.
    selected.sort()
    return dict(
        version=SUBSET_VERSION,
        policy="greedy_pairwise_then_canonical_hash",
        requested_limit=limit,
        original_pool_size=len(configs),
        meaningful_pool_size=len(unique),
        actual_pool_size=len(selected),
        indices=selected,
        config_ids=[config_id(configs[i]) for i in selected],
        covered_features=len(covered),
        total_features=len(total),
        aliases=aliases,
        rejected=rejected,
    )
