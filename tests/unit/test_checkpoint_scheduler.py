import numpy as np
from puredl.optim import SGD, CosineAnnealingLR
from puredl.training import save_checkpoint, load_checkpoint
from puredl.vision import MLPClassifier

def test_checkpoint_roundtrip(tmp_path):
    rng=np.random.default_rng(3); model=MLPClassifier(4,5,3,rng=rng); opt=SGD(model.parameters(),0.1,momentum=0.9); sched=CosineAnnealingLR(opt,10,0.001)
    original={k:v.copy() for k,v in model.state_dict().items()}; sched.step(); path=tmp_path/'state.npz'; save_checkpoint(path,model=model,optimizer=opt,scheduler=sched,epoch=7,metrics={'acc':0.5})
    for p in model.parameters(): p.data += 4
    state=load_checkpoint(path,model=model,optimizer=opt,scheduler=sched)
    assert state['epoch']==7 and state['metrics']['acc']==0.5
    for k,v in model.state_dict().items(): np.testing.assert_allclose(v,original[k])

def test_cosine_scheduler_decreases_lr():
    model=MLPClassifier(4,5,3,rng=np.random.default_rng(1)); opt=SGD(model.parameters(),0.1); sched=CosineAnnealingLR(opt,4,0.0); vals=[]
    for _ in range(5): vals.append(sched.step())
    assert vals[0] == 0.1 and vals[-1] == 0.0 and all(a>=b for a,b in zip(vals,vals[1:]))
