import copy
import hashlib
import pytest
from PIL import Image,ImageDraw
from scifigbench import render,prepare_model_payload,DataError

def digest(im):return hashlib.sha256(im.tobytes()).hexdigest()

def test_five_task_pixels_and_input_boundary(dataset,monkeypatch,tmp_path):
    root,rows=dataset;monkeypatch.chdir(tmp_path)
    rgb={'red':(255,0,0),'orange':(255,140,0),'yellow':(250,200,0),'green':(0,170,0),'blue':(0,90,255),'violet':(160,32,240)}
    for q in rows:
        with Image.open(root/q['image_path']) as original:expected=original.convert('RGB')
        t=q['task'];draw=ImageDraw.Draw(expected);w,h=expected.size;thick=max(5,round(min(w,h)/130))
        if t=='dense_content_perception':boxes,colors=q['boxes_xyxy'],q['colors']
        elif t=='node_level_degree_counting':boxes,colors=[q['red_box_xyxy']],['red']
        elif t in ('edge_level_verification','path_level_reachability'):boxes,colors=[q['red_box_xyxy'],q['blue_box_xyxy']],['red','blue']
        else:boxes,colors=[],[]
        for (x1,y1,x2,y2),c in zip(boxes,colors):draw.rectangle([max(0,x1-thick),max(0,y1-thick),min(w,x2+thick),min(h,y2+thick)],outline=rgb[c],width=thick)
        payload=prepare_model_payload(q,data_root=root)
        assert set(payload)=={'prompt','image'} and payload['prompt']==q['prompt']
        assert digest(payload['image'])==digest(expected)
        poison={**q,'answer':'SECRET','meta':{'node_ids':[999],'target_node_id':999},'annotations':{'private':True},'gt':'SECRET','path':[999]}
        changed=prepare_model_payload(poison,data_root=root)
        assert changed['prompt']==payload['prompt'] and digest(changed['image'])==digest(payload['image'])
        def backend(**kwargs):assert set(kwargs)=={'prompt','image'}
        backend(**changed)

def test_clipping_draw_order_missing_image_and_actual_size(dataset):
    root,rows=dataset;q=copy.deepcopy(rows[2]);q['red_box_xyxy']=[0,0,180,120];q['blue_box_xyxy']=[0,0,20,20]
    im=render(q,data_root=root)
    assert im.getpixel((0,0))==(0,90,255)
    q['image_size']=[181,120]
    with pytest.raises(DataError,match='dimensions'):render(q,data_root=root)
    (root/'images/F000001.jpg').unlink()
    with pytest.raises(FileNotFoundError,match='data_root'):render(rows[0],data_root=root)

def test_size_dependent_thickness_and_six_colors(dataset):
    root,rows=dataset;q=copy.deepcopy(rows[0]);q['image_size']=[1600,1300]
    q['boxes_xyxy']=[[50+i*220,100,180+i*220,200] for i in range(6)];q['colors']=['red','orange','yellow','green','blue','violet']
    Image.new('RGB',(1600,1300),'white').save(root/'images/F000001.jpg')
    im=render(q,data_root=root)
    assert im.getpixel((40,100))==(255,0,0)
    assert im.getpixel((1140,100))==(160,32,240)
