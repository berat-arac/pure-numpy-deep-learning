from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from .parameter import Parameter


class Module:
    """Minimal NumPy module base class with explicit parameters and buffers."""

    def __init__(self) -> None:
        self.training = True
        self._buffers: dict[str, np.ndarray] = {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def register_buffer(self, name: str, value: np.ndarray) -> None:
        if not name or "." in name:
            raise ValueError("buffer name must be a nonempty local attribute name")
        array = np.asarray(value)
        self._buffers[name] = array
        setattr(self, name, array)

    def _walk_modules(self) -> Iterator[tuple[str, Module]]:
        seen: set[int] = set()

        def visit(prefix: str, value: Any):
            if isinstance(value, Module):
                if id(value) in seen:
                    return
                seen.add(id(value))
                yield prefix, value
                for name, child in value.__dict__.items():
                    if name == "_buffers":
                        continue
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    yield from visit(child_prefix, child)
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    child_prefix = f"{prefix}.{index}" if prefix else str(index)
                    yield from visit(child_prefix, child)
            elif isinstance(value, dict):
                for name, child in value.items():
                    child_prefix = f"{prefix}.{name}" if prefix else str(name)
                    yield from visit(child_prefix, child)

        yield from visit("", self)

    def parameters(self) -> list[Parameter]:
        return [parameter for _, parameter in self.named_parameters()]

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()

        def visit(prefix: str, value: Any):
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    yield prefix, value
                return
            if isinstance(value, Module):
                for name, child in value.__dict__.items():
                    if name == "_buffers":
                        continue
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    yield from visit(child_prefix, child)
                return
            if isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    child_prefix = f"{prefix}.{index}" if prefix else str(index)
                    yield from visit(child_prefix, child)
                return
            if isinstance(value, dict):
                for name, child in value.items():
                    child_prefix = f"{prefix}.{name}" if prefix else str(name)
                    yield from visit(child_prefix, child)

        for name, value in self.__dict__.items():
            if name == "_buffers":
                continue
            yield from visit(name, value)

    def named_buffers(self) -> Iterator[tuple[str, np.ndarray]]:
        seen: set[int] = set()
        for prefix, module in self._walk_modules():
            for local_name, buffer in module._buffers.items():
                if id(buffer) in seen:
                    continue
                seen.add(id(buffer))
                name = f"{prefix}.{local_name}" if prefix else local_name
                yield name, buffer

    def state_dict(self) -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {}
        for name, parameter in self.named_parameters():
            state[name] = parameter.data.copy()
        for name, buffer in self.named_buffers():
            state[name] = buffer.copy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray], *, strict: bool = True) -> None:
        destinations: dict[str, np.ndarray] = {}
        for name, parameter in self.named_parameters():
            destinations[name] = parameter.data
        for name, buffer in self.named_buffers():
            destinations[name] = buffer

        missing = sorted(set(destinations) - set(state))
        unexpected = sorted(set(state) - set(destinations))
        if strict and (missing or unexpected):
            raise KeyError(f"state mismatch, missing={missing}, unexpected={unexpected}")

        for name, destination in destinations.items():
            if name not in state:
                continue
            source = np.asarray(state[name])
            if source.shape != destination.shape:
                raise ValueError(
                    f"shape mismatch for {name}: expected {destination.shape}, got {source.shape}"
                )
            destination[...] = source.astype(destination.dtype, copy=False)

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def _set_training(self, mode: bool) -> Module:
        for _, module in self._walk_modules():
            module.training = mode
        return self

    def train(self) -> Module:
        return self._set_training(True)

    def eval(self) -> Module:
        return self._set_training(False)
