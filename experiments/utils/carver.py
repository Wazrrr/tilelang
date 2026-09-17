"""Evaluate explicit template tiles through Carver's existing policy equations."""

from tilelang.tiletune.ranking import rank_records, select_top_k


def evaluate_tile(policy, tile, steps, threads):
    td = policy.compute_tile_dict(tile, steps)
    valid = td.valid and policy.check_tile_shape_isvalid(td)
    valid = valid and all(policy._assign_block_size(node, td, threads) is not None for node in policy.ordered_nodes)
    return dict(
        valid=bool(valid),
        score=float((td.traffic + 1) * td.num_wave) if valid else None,
        traffic_bytes=float(td.traffic),
        shared_bytes=int(td.smem_cost),
        waves=int(td.num_wave) if td.valid else None,
        blocks_per_sm=int(td.block_per_SM) if td.valid else None,
    )


def template_report(configs, evaluations, top_k, *, template, assumptions):
    records = []
    for index, (config, models) in enumerate(zip(configs, evaluations)):
        valid = all(model["valid"] for model in models)
        records.append(
            dict(
                index=index,
                config=dict(config),
                status="analyzed" if valid else "model_rejected",
                tile_cost=dict(score=sum(model["score"] for model in models) if valid else None, ranking_metric="carver_traffic_waves"),
                model=dict(valid=valid, components=models),
            )
        )
    ranking = rank_records(records)
    selected = select_top_k(ranking, top_k)
    for row in records:
        row["selected"] = row["index"] in selected
        if row["status"] == "analyzed" and not row["selected"]:
            row["status"] = "not_selected"
    return dict(
        model=template,
        metric="carver_traffic_waves",
        score_units="byte-waves",
        configs=records,
        ranking=ranking,
        selection=dict(requested_k=top_k, selected_indices=selected, selected_count=len(selected), shortfall=top_k - len(selected)),
        formula="sum of each template graph's (traffic_bytes + 1) * num_wave",
        assumptions=assumptions,
    )
