"""Dense content perception scoring."""
from __future__ import annotations
from ..parsing import parse_json_list
from .._latex_norm import anls, position_anls, set_anls

def score(q: dict, raw, args) -> dict:
    normalize = not args.no_latex_normalize
    gt = q["answer"]
    if not isinstance(gt, list):
        gt = [gt]
    pred = parse_json_list(raw)
    parse_ok = pred is not None
    pred = pred or []
    subtask = q.get("subtask")

    if subtask == "single":
        g0 = gt[0] if gt else ""
        p0 = pred[0] if pred else ""
        s = anls(g0, p0, normalize=normalize)
    else:  # flow_path (or any other list subtask): position-aligned
        s = position_anls(gt, pred, normalize=normalize)
    # Extra metric: keep the original single-OCR scoring semantics, and only
    # replace flow_path's position binding with content_task's Hungarian set
    # matching. Applying set_anls to single questions would penalize extra
    # strings even though the primary single metric only binds to pred[0].
    if subtask == "single":
        set_s = s
    else:
        set_s = set_anls([str(x) for x in gt], [str(x) for x in pred],
                         normalize=normalize)

    m = q.get("meta", {})
    return {
        "score": s,
        "set_score": set_s,
        "parse_ok": parse_ok,
        "pred_parsed": pred,
        "gt": gt,
        "stratum": {
            "subtask": subtask,
            "K": m.get("K"),
            "area_bucket": m.get("area_bucket"),
            "len_bucket": m.get("len_bucket"),
        },
    }
