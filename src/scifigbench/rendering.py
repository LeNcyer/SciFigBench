"""Canonical visual inputs: only the QA prompt and necessary rendered image."""
from __future__ import annotations
from pathlib import Path
import re
from PIL import Image, ImageDraw
from .tasks import TASKS
from .data import resolve_data_root, validate_render_fields, DataError
RGB={'red':(255,0,0),'orange':(255,140,0),'yellow':(250,200,0),'green':(0,170,0),'blue':(0,90,255),'violet':(160,32,240)}

def rendering_view(q):
    task=q['task'];assert task in TASKS
    keys=['task','image_path','image_size']
    if task=='dense_content_perception':keys+=['boxes_xyxy','colors']
    elif task in ('edge_level_verification','path_level_reachability'):keys+=['red_box_xyxy','blue_box_xyxy']
    elif task=='node_level_degree_counting':keys+=['red_box_xyxy']
    return {k:q[k] for k in keys}

def _render_image(q,image_root):
    v=rendering_view(q);rel=v['image_path'];assert re.fullmatch(r'images/F[0-9]{6}\.jpg',rel),rel
    path=Path(image_root)/rel
    with Image.open(path) as im:canvas=im.convert('RGB')
    w,h=canvas.size;assert list(canvas.size)==v['image_size']
    task=v['task']
    if task=='information_flow_tracing':return canvas
    thick=max(5,int(round(min(w,h)/130)));draw=ImageDraw.Draw(canvas)
    if task=='dense_content_perception':boxes=v['boxes_xyxy'];colors=v['colors'];assert len(boxes)==len(colors)
    elif task=='node_level_degree_counting':boxes=[v['red_box_xyxy']];colors=['red']
    else:boxes=[v['red_box_xyxy'],v['blue_box_xyxy']];colors=['red','blue']
    for b,c in zip(boxes,colors):
        x1,y1,x2,y2=b
        draw.rectangle([max(0,x1-thick),max(0,y1-thick),min(w,x2+thick),min(h,y2+thick)],outline=RGB[c],width=thick)
    return canvas

def render(qa: dict, *, data_root: str | Path | None = None) -> Image.Image:
    """Return a new canonical RGB image, without reading answers or metadata."""
    validate_render_fields(qa)
    root = resolve_data_root(data_root)
    try:
        return _render_image(qa, root)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image not found beneath data_root: {root / qa['image_path']}") from exc
    except AssertionError as exc:
        raise DataError("Actual image dimensions do not match QA image_size") from exc

def prepare_model_payload(qa: dict, *, data_root: str | Path | None = None) -> dict:
    """Only prompt and image are returned; no GT, IDs, paths or metadata."""
    if not isinstance(qa.get("prompt"), str) or not qa["prompt"].strip():
        raise DataError("QA prompt must be a nonempty string")
    return {"prompt": qa["prompt"], "image": render(qa, data_root=data_root)}
