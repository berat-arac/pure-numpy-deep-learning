from __future__ import annotations

import numpy as np

from puredl.core import Module, Parameter


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


class VanillaRNN(Module):
    """Tanh recurrent layer with explicit backpropagation through time."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        rng = rng or np.random.default_rng()
        self.weight_ih = Parameter(
            rng.normal(0.0, 1.0 / np.sqrt(input_size), size=(hidden_size, input_size)).astype(dtype)
        )
        self.weight_hh = Parameter(
            rng.normal(0.0, 1.0 / np.sqrt(hidden_size), size=(hidden_size, hidden_size)).astype(dtype)
        )
        self.bias = Parameter(np.zeros(hidden_size, dtype=dtype))
        self._x: np.ndarray | None = None
        self._h: np.ndarray | None = None

    def forward(self, x: np.ndarray, h0: np.ndarray | None = None) -> np.ndarray:
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError("x must have shape [batch, time, input_size]")
        batch, time, _ = x.shape
        if h0 is None:
            h0 = np.zeros((batch, self.hidden_size), dtype=x.dtype)
        if h0.shape != (batch, self.hidden_size):
            raise ValueError("h0 shape mismatch")

        h = np.empty((batch, time + 1, self.hidden_size), dtype=x.dtype)
        h[:, 0] = h0
        for t in range(time):
            pre = (
                x[:, t] @ self.weight_ih.data.T
                + h[:, t] @ self.weight_hh.data.T
                + self.bias.data
            )
            h[:, t + 1] = np.tanh(pre)
        self._x = x
        self._h = h
        return h[:, 1:]

    def backward(self, grad_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._x is None or self._h is None:
            raise RuntimeError("forward must be called before backward")
        x, h = self._x, self._h
        batch, time, _ = x.shape
        if grad_h.shape != (batch, time, self.hidden_size):
            raise ValueError("grad_h shape mismatch")

        self.weight_ih.grad.fill(0)
        self.weight_hh.grad.fill(0)
        self.bias.grad.fill(0)
        grad_x = np.zeros_like(x)
        grad_next = np.zeros((batch, self.hidden_size), dtype=x.dtype)

        for t in range(time - 1, -1, -1):
            total = grad_h[:, t] + grad_next
            current_h = h[:, t + 1]
            grad_pre = total * (1.0 - current_h * current_h)
            self.weight_ih.grad += grad_pre.T @ x[:, t]
            self.weight_hh.grad += grad_pre.T @ h[:, t]
            self.bias.grad += grad_pre.sum(axis=0)
            grad_x[:, t] = grad_pre @ self.weight_ih.data
            grad_next = grad_pre @ self.weight_hh.data
        return grad_x, grad_next

    def step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        return np.tanh(
            x_t @ self.weight_ih.data.T
            + h_prev @ self.weight_hh.data.T
            + self.bias.data
        )


class LSTM(Module):
    """LSTM recurrent layer with combined gates and explicit BPTT."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        rng = rng or np.random.default_rng()
        gate_size = 4 * hidden_size
        self.weight_ih = Parameter(
            rng.normal(0.0, 1.0 / np.sqrt(input_size), size=(gate_size, input_size)).astype(dtype)
        )
        self.weight_hh = Parameter(
            rng.normal(0.0, 1.0 / np.sqrt(hidden_size), size=(gate_size, hidden_size)).astype(dtype)
        )
        bias = np.zeros(gate_size, dtype=dtype)
        bias[hidden_size : 2 * hidden_size] = 1.0
        self.bias = Parameter(bias)
        self._x: np.ndarray | None = None
        self._h: np.ndarray | None = None
        self._c: np.ndarray | None = None
        self._i: np.ndarray | None = None
        self._f: np.ndarray | None = None
        self._g: np.ndarray | None = None
        self._o: np.ndarray | None = None

    def forward(
        self,
        x: np.ndarray,
        h0: np.ndarray | None = None,
        c0: np.ndarray | None = None,
    ) -> np.ndarray:
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError("x must have shape [batch, time, input_size]")
        batch, time, _ = x.shape
        if h0 is None:
            h0 = np.zeros((batch, self.hidden_size), dtype=x.dtype)
        if c0 is None:
            c0 = np.zeros((batch, self.hidden_size), dtype=x.dtype)
        expected = (batch, self.hidden_size)
        if h0.shape != expected or c0.shape != expected:
            raise ValueError("initial state shape mismatch")

        h = np.empty((batch, time + 1, self.hidden_size), dtype=x.dtype)
        c = np.empty_like(h)
        i_all = np.empty((batch, time, self.hidden_size), dtype=x.dtype)
        f_all = np.empty_like(i_all)
        g_all = np.empty_like(i_all)
        o_all = np.empty_like(i_all)
        h[:, 0] = h0
        c[:, 0] = c0
        H = self.hidden_size

        for t in range(time):
            gates = (
                x[:, t] @ self.weight_ih.data.T
                + h[:, t] @ self.weight_hh.data.T
                + self.bias.data
            )
            i = _sigmoid(gates[:, 0:H])
            f = _sigmoid(gates[:, H : 2 * H])
            g = np.tanh(gates[:, 2 * H : 3 * H])
            o = _sigmoid(gates[:, 3 * H : 4 * H])
            c[:, t + 1] = f * c[:, t] + i * g
            h[:, t + 1] = o * np.tanh(c[:, t + 1])
            i_all[:, t] = i
            f_all[:, t] = f
            g_all[:, t] = g
            o_all[:, t] = o

        self._x = x
        self._h = h
        self._c = c
        self._i = i_all
        self._f = f_all
        self._g = g_all
        self._o = o_all
        return h[:, 1:]

    def backward(self, grad_h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = (self._x, self._h, self._c, self._i, self._f, self._g, self._o)
        if any(item is None for item in cached):
            raise RuntimeError("forward must be called before backward")
        x = self._x
        h = self._h
        c = self._c
        i_all = self._i
        f_all = self._f
        g_all = self._g
        o_all = self._o
        assert x is not None and h is not None and c is not None
        assert i_all is not None and f_all is not None and g_all is not None and o_all is not None

        batch, time, _ = x.shape
        if grad_h.shape != (batch, time, self.hidden_size):
            raise ValueError("grad_h shape mismatch")

        self.weight_ih.grad.fill(0)
        self.weight_hh.grad.fill(0)
        self.bias.grad.fill(0)
        grad_x = np.zeros_like(x)
        grad_h_next = np.zeros((batch, self.hidden_size), dtype=x.dtype)
        grad_c_next = np.zeros_like(grad_h_next)

        for t in range(time - 1, -1, -1):
            i = i_all[:, t]
            f = f_all[:, t]
            g = g_all[:, t]
            o = o_all[:, t]
            c_now = c[:, t + 1]
            c_prev = c[:, t]

            total_h = grad_h[:, t] + grad_h_next
            tanh_c = np.tanh(c_now)
            grad_o = total_h * tanh_c
            grad_c = grad_c_next + total_h * o * (1.0 - tanh_c * tanh_c)
            grad_f = grad_c * c_prev
            grad_c_prev = grad_c * f
            grad_i = grad_c * g
            grad_g = grad_c * i

            grad_i_pre = grad_i * i * (1.0 - i)
            grad_f_pre = grad_f * f * (1.0 - f)
            grad_g_pre = grad_g * (1.0 - g * g)
            grad_o_pre = grad_o * o * (1.0 - o)
            grad_gates = np.concatenate(
                (grad_i_pre, grad_f_pre, grad_g_pre, grad_o_pre), axis=1
            )

            self.weight_ih.grad += grad_gates.T @ x[:, t]
            self.weight_hh.grad += grad_gates.T @ h[:, t]
            self.bias.grad += grad_gates.sum(axis=0)
            grad_x[:, t] = grad_gates @ self.weight_ih.data
            grad_h_next = grad_gates @ self.weight_hh.data
            grad_c_next = grad_c_prev

        return grad_x, grad_h_next, grad_c_next

    def step(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        H = self.hidden_size
        gates = (
            x_t @ self.weight_ih.data.T
            + h_prev @ self.weight_hh.data.T
            + self.bias.data
        )
        i = _sigmoid(gates[:, 0:H])
        f = _sigmoid(gates[:, H : 2 * H])
        g = np.tanh(gates[:, 2 * H : 3 * H])
        o = _sigmoid(gates[:, 3 * H : 4 * H])
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        return h, c
