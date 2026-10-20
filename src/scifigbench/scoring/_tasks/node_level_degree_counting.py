"""Node level degree counting scoring."""
from __future__ import annotations
from ..parsing import parse_int

def score(q: dict, raw, args) -> dict:
    gt = int(str(q["answer"]))
    pred = parse_int(raw)
    parse_ok = pred is not None
    s = 1.0 if (parse_ok and pred == gt) else 0.0
    m = q.get("meta", {})
    return {
        "score": s,
        "parse_ok": parse_ok,
        "pred_parsed": pred,
        "gt": gt,
        "stratum": {
            "variant": m.get("variant"),
            "degree_bucket": m.get("degree_bucket"),
        },
    }
