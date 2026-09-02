from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from puredl.core import Module, Parameter

def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int): return value, value
    if len(value) != 2: raise ValueError("Expected an int or a pair")
    return int(value[0]), int(value[1])

class Conv2D(Module):
    """NCHW Conv2D implemented with NumPy only.

    Supports kernel size, stride, symmetric zero padding, and optional bias.
    Dilation and grouped convolution are deferred to later vision checkpoints.
    """
    def __init__(self,in_channels,out_channels,kernel_size,*,stride=1,padding=0,bias=True,rng=None,dtype=np.float32):
        super().__init__(); kh,kw=_pair(kernel_size); sh,sw=_pair(stride); ph,pw=_pair(padding)
        if min(in_channels,out_channels,kh,kw,sh,sw)<=0: raise ValueError("Channels, kernel size, and stride must be positive")
        if min(ph,pw)<0: raise ValueError("Padding must be nonnegative")
        self.in_channels=in_channels; self.out_channels=out_channels; self.kernel_size=(kh,kw); self.stride=(sh,sw); self.padding=(ph,pw)
        rng=rng or np.random.default_rng(); fan_in=in_channels*kh*kw
        self.weight=Parameter(rng.normal(0,np.sqrt(2/fan_in),size=(out_channels,in_channels,kh,kw)).astype(dtype))
        self.bias=Parameter(np.zeros(out_channels,dtype=dtype)) if bias else None
        self._x_pad=None; self._windows=None; self._input_shape=None
    def _output_shape(self,h,w):
        kh,kw=self.kernel_size; sh,sw=self.stride; ph,pw=self.padding
        return (h+2*ph-kh)//sh+1,(w+2*pw-kw)//sw+1
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=self.in_channels: raise ValueError("Conv2D expects NCHW input with matching channels")
        ph,pw=self.padding; sh,sw=self.stride; kh,kw=self.kernel_size
        x_pad=np.pad(x,((0,0),(0,0),(ph,ph),(pw,pw)))
        windows=np.lib.stride_tricks.sliding_window_view(x_pad,(kh,kw),axis=(2,3))[:,:,::sh,::sw,:,:]
        out=np.einsum('ncyxkl,ockl->noyx',windows,self.weight.data,optimize=True)
        if self.bias is not None: out=out+self.bias.data[None,:,None,None]
        self._x_pad=x_pad; self._windows=windows; self._input_shape=x.shape
        return out
    def backward(self,grad_out):
        if self._windows is None or self._x_pad is None or self._input_shape is None: raise RuntimeError("forward must be called before backward")
        windows=self._windows; x_pad=self._x_pad; n,c,h,w=self._input_shape
        if grad_out.shape[:2]!=(n,self.out_channels) or grad_out.shape[2:4]!=windows.shape[2:4]: raise ValueError("grad_out shape mismatch")
        self.weight.grad[...] = np.einsum('noyx,ncyxkl->ockl',grad_out,windows,optimize=True)
        if self.bias is not None: self.bias.grad[...] = grad_out.sum(axis=(0,2,3))
        grad_x_pad=np.zeros_like(x_pad,dtype=np.result_type(x_pad,grad_out))
        kh,kw=self.kernel_size; sh,sw=self.stride; ph,pw=self.padding; oh,ow=grad_out.shape[2:]
        for ky in range(kh):
            ys=slice(ky,ky+sh*oh,sh)
            for kx in range(kw):
                xs=slice(kx,kx+sw*ow,sw)
                contrib=np.einsum('noyx,oc->ncyx',grad_out,self.weight.data[:,:,ky,kx],optimize=True)
                grad_x_pad[:,:,ys,xs] += contrib
        if ph==0 and pw==0: return grad_x_pad
        return grad_x_pad[:,:,ph:ph+h,pw:pw+w]
