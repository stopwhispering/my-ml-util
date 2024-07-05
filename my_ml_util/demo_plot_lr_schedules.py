import torch
import transformers
from torch.optim import lr_scheduler

from my_ml_util.visualize import plot_lr_schedule, plot_lr_schedule_with_loss

dummy_model = torch.nn.Linear(10, 1)
dummy_optimizer = torch.optim.AdamW(params=dummy_model.parameters(),
                                    lr=1e-3,
                                    )
scheduler = lr_scheduler.LinearLR(
    optimizer=dummy_optimizer,
    start_factor=1.0,
    end_factor=0.001,
    total_iters=5,  # number of steps it takes to reach end factor
)
plot_lr_schedule(scheduler, title="lr_scheduler.LinearLR")

scheduler = lr_scheduler.StepLR(
    optimizer=dummy_optimizer,
    step_size=2,  # Period of learning rate decay
    gamma=0.9,  # Multiplicative factor of learning rate decay
)
plot_lr_schedule(scheduler, title="lr_scheduler.StepLR")

scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer=dummy_optimizer,
    T_max=9,  # max. number of iterations; after that the lr will rise again!!!
    eta_min=0,  # Minimum learning rate
)
plot_lr_schedule(scheduler, title="lr_scheduler.CosineAnnealingLR")

# huggingface schedule helpers have a nicer API
# (returns torch.optim.lr_scheduler.LambdaLR)
# exactly same result as above
scheduler = transformers.get_cosine_schedule_with_warmup(
    optimizer=dummy_optimizer,
    num_warmup_steps=0,
    num_training_steps=9,
    )
plot_lr_schedule(scheduler, title="transformers.get_cosine_schedule_with_warmup w/o warmup")

# huggingface schedule helpers have a nicer API
# (returns torch.optim.lr_scheduler.LambdaLR)
# exactly same result as above
scheduler = transformers.get_cosine_schedule_with_warmup(
    optimizer=dummy_optimizer,
    num_warmup_steps=2,
    num_training_steps=9,
    )
plot_lr_schedule(scheduler, title="transformers.get_cosine_schedule_with_warmup")

# with ReduceLROnPlateau, the learning rate is dependent on the loss development
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer=dummy_optimizer,
    mode='min',  # lr is reduced when the quantity has stopped decreasing
    factor=0.1,  # factor by which the learning rate is reduced
    patience=1,  # default: 10
)
plot_lr_schedule_with_loss(scheduler,
                           title="lr_scheduler.ReduceLROnPlateau",
                           losses=[0.5]*10)

scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer=dummy_optimizer,
    mode='min',  # lr is reduced when the quantity has stopped decreasing
    factor=0.1,  # factor by which the learning rate is reduced
    patience=1,  # default: 10
)
plot_lr_schedule_with_loss(scheduler,
                           title="lr_scheduler.ReduceLROnPlateau",
                           losses=[0.6, 0.6,
                                   0.5, 0.5,
                                   0.4, 0.4,
                                   0.3, 0.3,
                                   0.2, 0.2]
                           )

scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer=dummy_optimizer,
    mode='min',  # lr is reduced when the quantity has stopped decreasing
    factor=0.9,  # factor by which the learning rate is reduced
    patience=1,  # default: 10
)
plot_lr_schedule_with_loss(scheduler,
                           title="lr_scheduler.ReduceLROnPlateau",
                           losses=[0.9, 0.8,
                                   0.7, 0.7,
                                   0.7, 0.6,
                                   0.5, 0.5,
                                   0.5, 0.6,
                                   0.5, 0.5,
                                   0.4, 0.4,
                                   0.3, 0.3]
                           )