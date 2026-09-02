from __future__ import annotations
import numpy as np
class Adam:
    def __init__(self,parameters,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0):
        self.parameters=list(parameters); self.lr=float(lr); self.beta1,self.beta2=map(float,betas); self.eps=float(eps); self.weight_decay=float(weight_decay)
        self._m=[np.zeros_like(p.data) for p in self.parameters]; self._v=[np.zeros_like(p.data) for p in self.parameters]; self._step=0
    def zero_grad(self):
        for p in self.parameters: p.zero_grad()
    def step(self):
        self._step+=1; b1,b2=self.beta1,self.beta2
        for p,m,v in zip(self.parameters,self._m,self._v,strict=True):
            g=p.grad + self.weight_decay*p.data if self.weight_decay else p.grad
            m*=b1; m+=(1-b1)*g; v*=b2; v+=(1-b2)*(g*g)
            p.data -= self.lr*(m/(1-b1**self._step))/(np.sqrt(v/(1-b2**self._step))+self.eps)
    def state_dict(self):
        return {"lr":self.lr,"beta1":self.beta1,"beta2":self.beta2,"eps":self.eps,"weight_decay":self.weight_decay,"step":self._step,"m":[x.copy() for x in self._m],"v":[x.copy() for x in self._v]}
    def load_state_dict(self,state):
        self.lr=float(state["lr"]); self.beta1=float(state["beta1"]); self.beta2=float(state["beta2"]); self.eps=float(state["eps"]); self.weight_decay=float(state.get("weight_decay",0)); self._step=int(state["step"])
        for dst,src in zip(self._m,state["m"],strict=True): dst[...] = src
        for dst,src in zip(self._v,state["v"],strict=True): dst[...] = src
