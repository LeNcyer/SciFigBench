"""Information flow tracing scoring."""
from __future__ import annotations
import json
from ..parsing import parse_json_list
from .._latex_norm import set_anls

def score(q: dict, raw, args) -> dict:
    normalize = not args.no_latex_normalize
    gt_raw = q["answer"]
    if isinstance(gt_raw, str):
        try:
            gt = json.loads(gt_raw)
            if not isinstance(gt, list):
                gt = [str(gt_raw)]
        except Exception:
            gt = [gt_raw]
    else:
        gt = list(gt_raw)
    pred = parse_json_list(raw)
    parse_ok = pred is not None
    pred = pred or []
    s = set_anls([str(x) for x in gt], [str(x) for x in pred],
                 normalize=normalize)
    m = q.get("meta", {})
    return {
        "score": s,
        "parse_ok": parse_ok,
        "pred_parsed": pred,
        "gt": gt,
        "stratum": {
            "variant": m.get("variant"),
            "target_type": m.get("target_type"),
            "n_gt_neighbors": len(gt),
        },
    }
