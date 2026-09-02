from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from puredl.core import Module, Parameter
from .conv import Conv2D, _pair


class DepthwiseConv2D(Module):
    """Depthwise NCHW convolution with one spatial kernel per input channel."""

    def __init__(
        self,
        channels: int,
        kernel_size: int | Sequence[int],
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        bias: bool = True,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        kh, kw = _pair(kernel_size)
        sh, sw = _pair(stride)
        ph, pw = _pair(padding)
        if min(channels, kh, kw, sh, sw) <= 0:
            raise ValueError("channels, kernel size, and stride must be positive")
        if min(ph, pw) < 0:
            raise ValueError("padding must be nonnegative")

        self.channels = int(channels)
        self.kernel_size = (kh, kw)
        self.stride = (sh, sw)
        self.padding = (ph, pw)
        rng = rng or np.random.default_rng()
        fan_in = kh * kw
        weight = rng.normal(
            0.0,
            np.sqrt(2.0 / fan_in),
            size=(channels, 1, kh, kw),
        ).astype(dtype)
        self.weight = Parameter(weight)
        self.bias = Parameter(np.zeros(channels, dtype=dtype)) if bias else None
        self._windows: np.ndarray | None = None
        self._x_pad: np.ndarray | None = None
        self._input_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("DepthwiseConv2D expects NCHW input with matching channels")
        ph, pw = self.padding
        sh, sw = self.stride
        kh, kw = self.kernel_size
        x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
        windows = np.lib.stride_tricks.sliding_window_view(
            x_pad, (kh, kw), axis=(2, 3)
        )[:, :, ::sh, ::sw, :, :]
        kernel = self.weight.data[:, 0]
        out = np.einsum("ncyxkl,ckl->ncyx", windows, kernel, optimize=True)
        if self.bias is not None:
            out = out + self.bias.data[None, :, None, None]
        self._windows = windows
        self._x_pad = x_pad
        self._input_shape = x.shape
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._windows is None or self._x_pad is None or self._input_shape is None:
            raise RuntimeError("forward must be called before backward")
        n, c, h, w = self._input_shape
        if grad_out.ndim != 4 or grad_out.shape[:2] != (n, c):
            raise ValueError("grad_out shape mismatch")
        windows = self._windows
        if grad_out.shape[2:4] != windows.shape[2:4]:
            raise ValueError("grad_out spatial shape mismatch")

        self.weight.grad[:, 0, :, :] = np.einsum(
            "ncyx,ncyxkl->ckl", grad_out, windows, optimize=True
        )
        if self.bias is not None:
            self.bias.grad[...] = grad_out.sum(axis=(0, 2, 3))

        grad_x_pad = np.zeros_like(
            self._x_pad, dtype=np.result_type(self._x_pad, grad_out)
        )
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oh, ow = grad_out.shape[2:]
        kernel = self.weight.data[:, 0]
        for ky in range(kh):
            ys = slice(ky, ky + sh * oh, sh)
            for kx in range(kw):
                xs = slice(kx, kx + sw * ow, sw)
                grad_x_pad[:, :, ys, xs] += (
                    grad_out * kernel[None, :, ky, kx, None, None]
                )

        if ph == 0 and pw == 0:
            return grad_x_pad
        return grad_x_pad[:, :, ph : ph + h, pw : pw + w]


class PointwiseConv2D(Conv2D):
    """Explicit 1x1 convolution used for channel mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int | Sequence[int] = 1,
        bias: bool = True,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=bias,
            rng=rng,
            dtype=dtype,
        )


class DepthwiseSeparableConv2D(Module):
    """Depthwise spatial convolution followed by a 1x1 pointwise convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 1,
        bias: bool = False,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.depthwise = DepthwiseConv2D(
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            rng=rng,
            dtype=dtype,
        )
        self.pointwise = PointwiseConv2D(
            in_channels,
            out_channels,
            bias=bias,
            rng=rng,
            dtype=dtype,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.pointwise.forward(self.depthwise.forward(x))

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return self.depthwise.backward(self.pointwise.backward(grad_out))
