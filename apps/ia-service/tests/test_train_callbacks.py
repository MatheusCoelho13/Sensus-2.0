import sys
import types
from pathlib import Path

for name in ('torch', 'ultralytics'):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules['ultralytics'].YOLO = object
# torch falso: só o que epoch_losses usa no caminho de tensor antigo
sys.modules['torch'].as_tensor = lambda x: types.SimpleNamespace(
    flatten=lambda: types.SimpleNamespace(tolist=lambda: list(x)))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
import train  # noqa: E402
import train_tui  # noqa: E402


class FakeScalar:  # tensor 0-d: float() funciona
    def __init__(self, v): self.v = v
    def __float__(self): return self.v


def test_epoch_losses_ultralytics_84_dict():
    # regressão: tloss é dict em Ultralytics >= 8.4 ('dict' object has no attribute 'tolist')
    t = types.SimpleNamespace(tloss={'box_loss': FakeScalar(1.5), 'cls_loss': 2.0}, loss_names=())
    assert train.epoch_losses(t) == {'box_loss': 1.5, 'cls_loss': 2.0}


def test_epoch_losses_old_tensor_and_none():
    t = types.SimpleNamespace(tloss=[1.0, 2.0, 3.0], loss_names=('box_loss', 'cls_loss', 'dfl_loss'))
    assert train.epoch_losses(t) == {'box_loss': 1.0, 'cls_loss': 2.0, 'dfl_loss': 3.0}
    assert train.epoch_losses(types.SimpleNamespace(tloss=None)) == {}


def test_callback_error_does_not_kill_training(capsys):
    @train.safe
    def boom(trainer):
        raise KeyError('x')
    boom(object())   # não levanta
    assert 'boom falhou' in capsys.readouterr().out


def test_upscale_smooth_and_keeps_constant():
    flat = [[[10, 20, 30]] * 3 for _ in range(3)]
    assert all(p == (10, 20, 30) for row in train_tui.upscale(flat, 3, 6) for p in row)
    grad = [[[0, 0, 0], [100, 100, 100], [200, 200, 200]]] * 3
    row = [p[0] for p in train_tui.upscale(grad, 3, 8)[0]]
    assert row == sorted(row) and row[0] == 0 and row[-1] == 200 and len(set(row)) > 3   # degradê suave
