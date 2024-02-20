import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager, ExitStack

import math
import numpy as np
import matplotlib.pyplot as plt


def get_runtime():
    if 'get_ipython' in locals() and 'runtime' in get_ipython().config.IPKernelApp.connection_file:  # noqa
        return 'kaggle'  # kaggle interactive
    if '/kaggle/input/' in __file__ or '/opt/conda/lib/' in __file__:
        return 'kaggle'  # kaggle from dataset script
    if 'get_ipython' not in locals():
        return 'script'  # local script
    if 'Roaming' in get_ipython().config.IPKernelApp.connection_file:  # noqa
        return 'local'
    return 'background'  # kaggle in background (save & run or submit to competition)


@contextmanager
def suppress_output(out=True, err=False):
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield



class LossPlot:
    def __init__(self, n_iterations: int, update_after_n_iterations=10):
        self.n_iterations = n_iterations
        self.n_updates = math.ceil(n_iterations / update_after_n_iterations)
        self.update_after_n_iterations = update_after_n_iterations

        self.current_iteration = 0
        self.x_axis_values = np.arange(self.n_iterations)

        self.train_losses = np.zeros(self.n_iterations)
        self.train_losses[:] = np.nan
        self.validation_losses = np.zeros(self.n_iterations)
        self.validation_losses[:] = np.nan

        self.fig = plt.figure()
        self.ax = plt.gca()
        self.ax.set_xlim([0, self.n_iterations])
        self.ax.set_ylim([0, 1])
        self.display_handle = ipython_display.display(self.fig,
                                                      display_id=True)

        self.train_line = None
        self.validation_line = None

    def update(self, train_loss: float, validation_loss: float | None = None):
        """generate the plot and update display handle to refresh it"""
        self.train_losses[self.current_iteration] = train_loss
        self.validation_losses[self.current_iteration] = validation_loss

        if self.current_iteration % self.update_after_n_iterations == 0 or self.current_iteration == self.n_iterations - 1:

            if self.train_line:
                self.train_line.set_xdata(self.x_axis_values)
                self.train_line.set_ydata(self.train_losses)
            else:
                self.train_line = self.ax.plot(self.x_axis_values,
                                               self.train_losses,
                                               color='blue',
                                               label='Train')[0]
                self.ax.legend()

            if validation_loss:
                if self.validation_line:
                    self.validation_line.set_xdata(self.x_axis_values)
                    self.validation_line.set_ydata(self.validation_losses)
                else:
                    self.validation_line = self.ax.plot(self.x_axis_values,
                                                        self.validation_losses,
                                                        color='red',
                                                        label='Validation')[0]
                    self.ax.legend()

            self.display_handle.update(self.fig)

        self.current_iteration += 1