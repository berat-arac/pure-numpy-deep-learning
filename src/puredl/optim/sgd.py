from __future__ import annotations
import numpy as np
from puredl.core import Parameter
class SGD:
    def __init__(self,parameters,lr,momentum=0.0,weight_decay=0.0):
        if lr<=0: raise ValueError("lr must be positive")
        if not 0<=momentum<1: raise ValueError("momentum must be in [0, 1)")
        self.parameters=list(parameters); self.lr=float(lr); self.momentum=float(momentum); self.weight_decay=float(weight_decay)
        self._velocity=[np.zeros_like(p.data) for p in self.parameters]
    def zero_grad(self):
        for p in self.parameters: p.zero_grad()
    def step(self):
        for p,v in zip(self.parameters,self._velocity,strict=True):
            g=p.grad + self.weight_decay*p.data if self.weight_decay else p.grad
            if self.momentum: v*=self.momentum; v+=g; g=v
            p.data -= self.lr*g
    def state_dict(self):
        return {"lr":self.lr,"momentum":self.momentum,"weight_decay":self.weight_decay,"velocity":[v.copy() for v in self._velocity]}
    def load_state_dict(self,state):
        self.lr=float(state["lr"]); self.momentum=float(state["momentum"]); self.weight_decay=float(state.get("weight_decay",0.0))
        vals=state["velocity"]
        if len(vals)!=len(self._velocity): raise ValueError("optimizer state parameter count mismatch")
        for dst,src in zip(self._velocity,vals,strict=True): dst[...] = src
