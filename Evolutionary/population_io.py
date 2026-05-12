"""
Save and load BaseUnit populations to / from JSON.

Each saved file captures:
- Population-level config: nd, num_op_obj, num_update_obj, dataset_type, source label.
- Per-unit: the five lookup tables, the w index, optional metadata (lift_variant,
  parent indices, last evaluated performance).

Reconstruction builds a fresh BaseUnit class via `base_unit_factory(nd, ...)` and
attaches each unit's tables as instance attributes (matching the post-mutation
shape used by `simple_mutate` / `splice_units`).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from Evolutionary.abstract_unit import AbstractBaseUnit, Mat
from Evolutionary.base_unit import base_unit_factory


TABLE_NAMES = [
    "FORWARD_TABLE",
    "FIND_BACK_INFO_TABLE",
    "BACKWARD_TABLE",
    "FIND_UPDATE_TABLE",
    "APPLY_UPDATE_TABLE",
]


def serialize_unit(unit: AbstractBaseUnit) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "w_idx": int(unit.w.w_idx),
        "tables": {name: getattr(unit, name) for name in TABLE_NAMES},
    }
    for tag in ("lift_variant", "lift_parents", "lift_permutation", "performance"):
        if hasattr(unit, tag):
            record[tag] = getattr(unit, tag)
    return record


def serialize_population(
    units: List[AbstractBaseUnit],
    *,
    num_data_obj: int,
    num_op_obj: int,
    num_update_obj: int,
    dataset_type: str,
    source: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "config": {
            "num_data_obj": num_data_obj,
            "num_op_obj": num_op_obj,
            "num_update_obj": num_update_obj,
            "dataset_type": dataset_type,
            "source": source,
            **(extra or {}),
        },
        "units": [serialize_unit(u) for u in units],
    }


def save_population(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def load_population(path: str) -> Dict[str, Any]:
    """Returns {'config': {...}, 'units': [BaseUnit, ...]}.

    A single fresh BaseUnit class is built and shared by every reconstructed
    instance — but per-unit tables are attached as instance attributes, so
    units do not share table state at runtime.
    """
    with open(path) as f:
        payload = json.load(f)
    cfg = payload["config"]
    nd = cfg["num_data_obj"]
    n_op = cfg.get("num_op_obj")
    n_upd = cfg.get("num_update_obj")
    base_cls = base_unit_factory(nd, num_mat=n_op, num_update_obj=n_upd)

    units = []
    for rec in payload["units"]:
        unit = base_cls(Mat(int(rec["w_idx"])))
        for name in TABLE_NAMES:
            setattr(unit, name, rec["tables"][name])
        for tag in ("lift_variant", "lift_parents", "lift_permutation", "performance"):
            if tag in rec:
                setattr(unit, tag, rec[tag])
        units.append(unit)
    return {"config": cfg, "units": units}
