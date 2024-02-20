import statistics
from contextlib import nullcontext
from time import time
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics

from my_ml_util.training import EarlyStopper


def train_or_validate_one_epoch_sample_fn(model,
                                          dataloader: DataLoader,
                                          criterion,
                                          optimizer,
                                          validate: bool,
                                          epoch: int,
                                          progress_bar,
                                          show_progress_bar: bool,
                                          update_progress_bar_fn: callable,
                                          ):
    current_epoch_losses = []
    current_epoch_scores = []
    current_epoch_targets_list = []
    current_epoch_outputs_list = []

    for i, batch_data in enumerate(progress_bar):

        x_batch, y_batch = batch_data

        # forward pass
        with torch.no_grad() if validate else nullcontext():

            pred = model(x_batch)  # [batch_size, n_classes]

            optimizer.zero_grad()
            loss = criterion(pred, y_batch)

        # backward pass
        if not validate:
            loss.backward()
            optimizer.step()

        # update progress bar
        current_epoch_losses.append(loss.item())
        current_epoch_targets_list.append(y_batch)
        current_epoch_outputs_list.append(pred)

        all_outputs = torch.cat(current_epoch_outputs_list)  # [n, n_classes]
        all_targets = torch.cat(current_epoch_targets_list)  # [n]
        current_epoch_mean_loss = statistics.mean(current_epoch_losses)

        predicted_classes = all_outputs.max(1).indices  # [n]), torch.int64
        score_current_batch = metrics.accuracy_score(y_true=all_targets.cpu().numpy(),
                                                     y_pred=predicted_classes.detach().cpu().numpy(), )

        current_epoch_scores.append(score_current_batch)
        current_epoch_mean_score = statistics.mean(current_epoch_scores)

        if show_progress_bar:
            update_progress_bar_fn(progress_bar=progress_bar,
                                   validate=validate,
                                   is_last_batch=(i == len(dataloader) - 1),
                                   epoch=epoch,
                                   loss=loss,
                                   current_epoch_mean_loss=current_epoch_mean_loss,
                                   current_epoch_mean_score=current_epoch_mean_score,
                                   )

    return current_epoch_mean_loss, current_epoch_mean_score  # noqa


class Trainer:
    def __init__(self,
                 criterion,
                 optimizer,
                 scheduler,
                 max_epochs: int,
                 early_stopper_patience: int,  # set to -1 to disable EarlyStopper
                 save_checkpoint: bool,
                 show_progress_bar: bool,
                 train_or_validate_one_epoch_custom_fn: Optional[Callable] = None,
                 max_duration_in_seconds: int = 9 * 60 * 60,
                 wandb=None,
                 ):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.early_stopper_patience = early_stopper_patience
        self.save_checkpoint = save_checkpoint
        self.show_progress_bar = show_progress_bar
        self.max_duration_in_seconds = max_duration_in_seconds
        self.wandb = wandb

        if self.wandb:
            self.wandb.define_metric("val/val_score", summary="max")
            self.wandb.define_metric("val/best_score", summary="max")
            self.wandb.define_metric("val/val_loss", summary="min")

        if train_or_validate_one_epoch_custom_fn is None:
            print('Using sample train_or_validate_one_epoch function. Please supply custom fn.')
            self.train_or_validate_one_epoch_custom_fn = train_or_validate_one_epoch_sample_fn
        else:
            self.train_or_validate_one_epoch_custom_fn = train_or_validate_one_epoch_custom_fn

    def fit(self,
            model,
            dataloader_train: DataLoader,
            dataloader_val: DataLoader,
            ):

        train_losses = []
        train_scores = []
        val_losses = []
        val_scores = []
        lrs = []
        early_stopped = False

        if self.early_stopper_patience != -1:
            early_stopper = EarlyStopper(patience=self.early_stopper_patience)
        epoch_saved_checkpoint = 0
        saved_checkpoint_score = 0

        start_time = time()
        for epoch in range(self.max_epochs):

            if self.scheduler:
                lrs.append(self.optimizer.param_groups[0]["lr"])

            train_mean_loss, train_mean_score = self.train_or_validate_one_epoch_wrapper(
                dataloader_train,
                model,
                validate=False,
                epoch=epoch,
            )
            train_losses.append(train_mean_loss)
            train_scores.append(train_mean_score)

            val_mean_loss, val_mean_score = self.train_or_validate_one_epoch_wrapper(
                dataloader_val,
                model,
                validate=True,
                epoch=epoch,
            )

            if self.wandb is not None:
                self.wandb.log({
                    "epoch": epoch,
                    "train/train_score": train_mean_score,
                    "train/train_loss": train_mean_loss,
                    "val/val_score": val_mean_score,
                    "val/val_loss": val_mean_loss,
                })

            if (not val_scores or val_mean_score >= max(val_scores)) and self.early_stopper_patience != -1:
                if self.save_checkpoint:
                    torch.save(model.state_dict(), 'best_model_state_dict.torch')
                epoch_saved_checkpoint = epoch
                saved_checkpoint_score = val_mean_score

                if self.wandb is not None:
                    self.wandb.run.summary["val/best_score"] = saved_checkpoint_score
                    self.wandb.run.summary["val/best_epoch"] = epoch_saved_checkpoint

            val_losses.append(val_mean_loss)
            val_scores.append(val_mean_score)

            if self.scheduler:
                self.scheduler.step(val_mean_loss)

            if self.early_stopper_patience != -1:
                early_stopper(-val_mean_score)

            seconds_elapsed = time() - start_time
            if seconds_elapsed > self.max_duration_in_seconds:
                print(f'Stopping after {seconds_elapsed} seconds.')
                break

            if self.early_stopper_patience != -1 and early_stopper.early_stop:
                print(f'Early stopping after epoch {epoch} w/ val score {val_mean_score:.4f}')
                early_stopped = True
                break

        if self.wandb is not None:
            self.wandb.run.summary["n_epochs_trained"] = epoch + 1  # noqa
            self.wandb.run.summary["early_stopped"] = early_stopped

        if self.early_stopper_patience == -1 and self.save_checkpoint:
            epoch_saved_checkpoint = epoch
            torch.save(model.state_dict(), 'final_model_state_dict.torch')

        return saved_checkpoint_score, epoch_saved_checkpoint, train_losses, val_losses, train_scores, val_scores, lrs

    def train_or_validate_one_epoch_wrapper(self,
                                            dataloader: DataLoader,
                                            model,
                                            validate: bool,
                                            epoch: int,
                                            ):
        if validate:
            model.eval()
        else:
            model.train()

        if self.show_progress_bar:
            progress_bar = tqdm(dataloader, total=len(dataloader))
        else:
            progress_bar = dataloader

        current_epoch_mean_loss, current_epoch_mean_score = self.train_or_validate_one_epoch_custom_fn(
            model=model,
            dataloader=dataloader,
            validate=validate,
            epoch=epoch,
            progress_bar=progress_bar,
            criterion=self.criterion,
            optimizer=self.optimizer,
            show_progress_bar=self.show_progress_bar,
            update_progress_bar_fn=self.update_progress_bar,
        )
        return current_epoch_mean_loss, current_epoch_mean_score  # noqa

    def update_progress_bar(self,
                            progress_bar,
                            validate: bool,
                            is_last_batch: bool,
                            epoch: int,
                            loss: torch.Tensor,
                            current_epoch_mean_loss: float,
                            current_epoch_mean_score: float,
                            ):

        mode = 'Validating' if validate else 'Training'

        current_lr = self.optimizer.param_groups[0]['lr']

        if not is_last_batch:
            progress_bar.set_description(
                f"{epoch} {mode}. Loss: {loss.item() :.4f} ({current_epoch_mean_loss:.4f}) | lr: {current_lr :.5f}")
        else:
            progress_bar.set_description(
                f"{epoch} {mode} | Avg. Loss: {current_epoch_mean_loss:.4f} ACC: {current_epoch_mean_score:.2f} [lr: {current_lr :.7f}]")
