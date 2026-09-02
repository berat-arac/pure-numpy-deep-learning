import numpy as np
from puredl.losses import CrossEntropyLoss
from puredl.vision import DeepCNN

def test_deep_cnn_forward_backward_shapes():
    rng=np.random.default_rng(2); model=DeepCNN(width=0.25,rng=rng); x=rng.normal(size=(2,3,32,32)).astype(np.float32); y=np.array([1,2]); loss=CrossEntropyLoss(); logits=model.forward(x); assert logits.shape==(2,10); loss.forward(logits,y); dx=model.backward(loss.backward()); assert dx.shape==x.shape
