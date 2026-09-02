from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def _encode(obj, arrays, prefix="root"):
    if isinstance(obj,np.ndarray):
        key=f"arr_{len(arrays)}"; arrays[key]=obj; return {"__array__":key}
    if isinstance(obj,(np.integer,np.floating)): return obj.item()
    if isinstance(obj,dict): return {str(k):_encode(v,arrays,f"{prefix}.{k}") for k,v in obj.items()}
    if isinstance(obj,list): return {"__list__":[_encode(v,arrays,prefix) for v in obj]}
    if isinstance(obj,tuple): return {"__tuple__":[_encode(v,arrays,prefix) for v in obj]}
    if obj is None or isinstance(obj,(str,int,float,bool)): return obj
    raise TypeError(f"unsupported checkpoint type: {type(obj).__name__}")

def _decode(obj, data):
    if isinstance(obj,dict) and "__array__" in obj: return data[obj["__array__"]].copy()
    if isinstance(obj,dict) and "__list__" in obj: return [_decode(v,data) for v in obj["__list__"]]
    if isinstance(obj,dict) and "__tuple__" in obj: return tuple(_decode(v,data) for v in obj["__tuple__"])
    if isinstance(obj,dict): return {k:_decode(v,data) for k,v in obj.items()}
    return obj

def save_checkpoint(path, *, model, optimizer=None, scheduler=None, epoch=0, metrics=None):
    state={"epoch":int(epoch),"model":model.state_dict(),"metrics":metrics or {}}
    if optimizer is not None: state["optimizer"]=optimizer.state_dict()
    if scheduler is not None: state["scheduler"]=scheduler.state_dict()
    arrays={}; meta=_encode(state,arrays)
    arrays["__metadata__"]=np.array(json.dumps(meta))
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**arrays)

def load_checkpoint(path, *, model, optimizer=None, scheduler=None, strict=True):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data["__metadata__"].item())); state=_decode(meta,data)
    model.load_state_dict(state["model"],strict=strict)
    if optimizer is not None and "optimizer" in state: optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state: scheduler.load_state_dict(state["scheduler"])
    return state
