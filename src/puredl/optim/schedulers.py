from __future__ import annotations
import math

class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer=optimizer; self.last_epoch=-1; self.base_lr=float(optimizer.lr)
    def get_lr(self): raise NotImplementedError
    def step(self):
        self.last_epoch += 1
        self.optimizer.lr=float(self.get_lr())
        return self.optimizer.lr
    def state_dict(self): return {"last_epoch":self.last_epoch,"base_lr":self.base_lr}
    def load_state_dict(self,state): self.last_epoch=int(state["last_epoch"]); self.base_lr=float(state["base_lr"]); self.optimizer.lr=float(self.get_lr())

class StepLR(LRScheduler):
    def __init__(self,optimizer,step_size,gamma=0.1):
        if step_size<=0: raise ValueError("step_size must be positive")
        self.step_size=int(step_size); self.gamma=float(gamma); super().__init__(optimizer)
    def get_lr(self):
        e=max(self.last_epoch,0); return self.base_lr*(self.gamma**(e//self.step_size))
    def state_dict(self): return {**super().state_dict(),"step_size":self.step_size,"gamma":self.gamma}
    def load_state_dict(self,state): self.step_size=int(state["step_size"]); self.gamma=float(state["gamma"]); super().load_state_dict(state)

class CosineAnnealingLR(LRScheduler):
    def __init__(self,optimizer,T_max,eta_min=0.0):
        if T_max<=0: raise ValueError("T_max must be positive")
        self.T_max=int(T_max); self.eta_min=float(eta_min); super().__init__(optimizer)
    def get_lr(self):
        t=min(max(self.last_epoch,0),self.T_max)
        return self.eta_min + 0.5*(self.base_lr-self.eta_min)*(1+math.cos(math.pi*t/self.T_max))
    def state_dict(self): return {**super().state_dict(),"T_max":self.T_max,"eta_min":self.eta_min}
    def load_state_dict(self,state): self.T_max=int(state["T_max"]); self.eta_min=float(state["eta_min"]); super().load_state_dict(state)
