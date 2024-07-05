from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    import torch.nn as nn


def display_trained_parameters(model: nn.Module) -> None:
    n_parameters = sum(p.numel() for p in model.parameters())
    n_trained_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{n_parameters = :_}")
    print(f"{n_trained_parameters = :_}")