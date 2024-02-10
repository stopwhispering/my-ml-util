from icecream import ic


class EarlyStopper:
    def __init__(self, patience=2):
        self.patience = patience
        self.validation_losses = []
        self.early_stop = False

    def __call__(self, validation_loss):
        self.validation_losses.append(validation_loss)

        if len(self.validation_losses) == 1:
            return

        # cancel if model has not improved in comparison to best epoch for more than patience epochs
        min_loss = min(self.validation_losses)

        counter_not_improved = 0
        for i, loss in enumerate(reversed(self.validation_losses)):
            if loss > min_loss:
                counter_not_improved += 1
            else:
                break

        self.early_stop = counter_not_improved > self.patience


if __name__ == '__main__':
    early_stopper = EarlyStopper(patience=1)
    early_stopper(5.0)
    assert not early_stopper.early_stop
    early_stopper(5.6)
    assert not early_stopper.early_stop
    early_stopper(5.7)
    assert early_stopper.early_stop


    early_stopper = EarlyStopper(patience=1)
    early_stopper(5.0)
    assert not early_stopper.early_stop
    early_stopper(5.6)
    assert not early_stopper.early_stop
    early_stopper(4.9)
    assert not early_stopper.early_stop
    early_stopper(5.4)
    assert not early_stopper.early_stop
    early_stopper(5.0)
    assert early_stopper.early_stop