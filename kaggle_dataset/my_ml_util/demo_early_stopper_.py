# from my_ml_util.training import EarlyStopper
from my_ml_util.training.early_stopper import EarlyStopper

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