"""Path level reachability scoring."""
from __future__ import annotations
from ..parsing import parse_bool

def score(q: dict, raw, args) -> dict:
    gt = str(q["answer"]).lower()
    pred = parse_bool(raw)
    parse_ok = pred is not None
    s = 1.0 if (parse_ok and pred == gt) else 0.0
    m = q.get("meta", {})
    return {
        "score": s,
        "parse_ok": parse_ok,
        "pred_parsed": pred,
        "gt": gt,
        "stratum": {
            "label": m.get("label"),
            "neg_subtype": m.get("neg_subtype"),
            "path_len_bucket": m.get("path_len_bucket"),
        },
    }
