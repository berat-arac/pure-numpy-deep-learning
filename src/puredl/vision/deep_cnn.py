from __future__ import annotations
import numpy as np
from puredl.core import Module, Sequential
from puredl.layers import Conv2D, BatchNorm2D, ReLU, MaxPool2D, GlobalAveragePool2D, Linear

class ConvBNReLU(Sequential):
    def __init__(self,in_channels,out_channels,*,rng,dtype=np.float32):
        super().__init__(
            Conv2D(in_channels,out_channels,3,padding=1,bias=False,rng=rng,dtype=dtype),
            BatchNorm2D(out_channels,dtype=dtype),
            ReLU(),
        )

class DeepCNN(Module):
    """CIFAR sized CNN with three stages and global average pooling."""
    def __init__(self,num_classes=10,*,width=1.0,rng=None,dtype=np.float32):
        super().__init__(); rng=rng or np.random.default_rng()
        c1=max(8,int(32*width)); c2=max(16,int(64*width)); c3=max(24,int(128*width))
        self.features=Sequential(
            ConvBNReLU(3,c1,rng=rng,dtype=dtype),
            ConvBNReLU(c1,c1,rng=rng,dtype=dtype),
            MaxPool2D(2),
            ConvBNReLU(c1,c2,rng=rng,dtype=dtype),
            ConvBNReLU(c2,c2,rng=rng,dtype=dtype),
            MaxPool2D(2),
            ConvBNReLU(c2,c3,rng=rng,dtype=dtype),
            MaxPool2D(2),
        )
        self.pool=GlobalAveragePool2D()
        self.classifier=Linear(c3,num_classes,rng=rng,dtype=dtype)
    def forward(self,x):
        x=self.features.forward(x); x=self.pool.forward(x); return self.classifier.forward(x)
    def backward(self,grad_out):
        g=self.classifier.backward(grad_out); g=self.pool.backward(g); return self.features.backward(g)
