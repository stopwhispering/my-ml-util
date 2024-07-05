from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class plot_backend(AbstractContextManager):
    """Context manager for switching matplotlib backend.
    Example: with plot_backend("agg"): ... to have a non-interactive backend."""

    backend: str
    current_backend: str

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.current_backend = plt.get_backend()
        available_backends = plt.rcsetup.all_backends
        assert backend in available_backends

    def __enter__(self):
        plt.switch_backend(self.backend)

    def __exit__(self, *args):
        plt.switch_backend(self.current_backend)


def plot_lr_schedule(scheduler: LRScheduler, title: str | None = None) -> None:
    lrs = []
    for i in range(10):
        # print(scheduler.get_last_lr()[0])
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.optimizer.step()
        scheduler.step()

    _fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_yticks(list(set(lrs)))
    df_lrs = pd.DataFrame({'LR': lrs,
                           'step': range(len(lrs))})
    sns.lineplot(data=df_lrs,
                 x='step',
                 y='LR',
                 ax=ax, )

    if title:
        ax.set_title(title)
    plt.show()


def plot_lr_schedule_with_loss(scheduler: ReduceLROnPlateau,
                               losses: list[float],
                               title: str | None = None,) -> None:
    """ReduceLROnPlateau has a different API than LRScheduler. Therefore we need a
    separate function to plot the learning rate schedule."""
    lrs = []
    for i in range(len(losses)):
        # print(scheduler.get_last_lr()[0])
        # lrs.append(scheduler.get_last_lr()[0])  # other API...
        lrs.append(scheduler.optimizer.param_groups[0]['lr'])
        scheduler.optimizer.step()
        scheduler.step(losses[i])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax2 = ax.twinx()

    ax.set_yticks(list(set(lrs)))
    df_lrs = pd.DataFrame({'LR': lrs,
                           'loss': losses,
                           'step': range(len(lrs))})
    sns.lineplot(data=df_lrs,
                 x='step',
                 y='LR',
                 ax=ax,
                 label='LR',
                 color='blue',
                 legend=False)

    sns.lineplot(data=df_lrs,
                 x='step',
                 y='loss',
                 ax=ax2,
                 label='loss',
                 color='red',
                 legend=False)

    fig.legend()
    if title:
        ax.set_title(title)
    plt.show()
