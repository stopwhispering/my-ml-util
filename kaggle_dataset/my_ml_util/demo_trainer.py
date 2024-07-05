import torch

from sklearn import datasets
from torch.utils.data import TensorDataset, DataLoader

from my_ml_util.training.trainer import Trainer

# from my_ml_util.training import Trainer

ds = datasets.load_iris()
x = torch.Tensor(ds['data'])
y = torch.Tensor(ds['target']).long()
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = torch.nn.Sequential(
    torch.nn.Linear(4, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 3),
)
Trainer(criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW(model.parameters()),
        scheduler=None,
        max_epochs=20,
        early_stopper_patience=3,
        save_checkpoint=False,
        show_progress_bar=True,
        max_duration_in_seconds=60).fit(
    model=model,
    dataloader_train=dataloader,
    dataloader_val=dataloader
)