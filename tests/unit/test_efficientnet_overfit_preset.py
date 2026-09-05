from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_training_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_efficientnet.py"
    spec = importlib.util.spec_from_file_location("train_efficientnet_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_overfit_preset_disables_regularization_and_uses_fixed_adam():
    module = _load_training_script()
    args = argparse.Namespace(
        dropout=0.2,
        drop_path=0.2,
        weight_decay=5e-4,
        no_augment=False,
        fixed_lr=False,
        optimizer="sgd",
        lr=0.025,
        limit_train=0,
        limit_test=0,
        checkpoint=None,
    )

    result = module.apply_overfit_preset(args)

    assert result.dropout == 0.0
    assert result.drop_path == 0.0
    assert result.weight_decay == 0.0
    assert result.no_augment is True
    assert result.fixed_lr is True
    assert result.optimizer == "adam"
    assert result.lr == 1e-3
    assert result.limit_train == 256
    assert result.limit_test == 256
    assert result.checkpoint.endswith("cifar10_efficientnet_overfit.npz")
