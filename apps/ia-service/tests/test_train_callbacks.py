import sys
import tempfile
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


def test_ckpt_info_reads_last_stat_from_pt():
    sys.modules['torch'].load = lambda p, **kw: {'train_metrics': {'metrics/mAP50(B)': 0.52, 'fitness': 0.3}}
    assert train.ckpt_info('models/best.pt') == {'map50': 0.52}
    sys.modules['torch'].load = lambda p, **kw: {'model': 'sem train_metrics (pesos oficiais)'}
    assert train.ckpt_info('yolov8n.pt') is None
    sys.modules['torch'].load = lambda p, **kw: (_ for _ in ()).throw(OSError('arquivo ruim'))
    assert train.ckpt_info('x.pt') is None    # nunca derruba o treino


def test_stop_reason_and_plain_explanation():
    assert train.stop_reason(True, False, 5, 100) == 'stopped'
    assert train.stop_reason(False, True, 5, 100) == 'goal'
    assert train.stop_reason(False, False, 100, 100) == 'epochs'
    assert train.stop_reason(False, False, 37, 100) == 'patience'
    assert '37 de 100' in train.explain_stop('patience', 37, 100, 30, 0.6) and 'patience' in train.explain_stop('patience', 37, 100, 30, 0.6)
    assert '100 épocas pedidas' in train.explain_stop('epochs', 100, 100, 30, 0.6)


def _fit_end(baseline, map50):
    train.STOP.update(asked=False, goal=False)
    cb = train.make_callbacks(0.60, None, '', baseline=baseline, patience=100)['on_fit_epoch_end']
    tr = types.SimpleNamespace(metrics={'metrics/mAP50(B)': map50}, epoch=0, epochs=300, stop=False, save_dir=tempfile.mkdtemp(), loss_names=())
    cb(tr)
    return tr.stop


def test_target_already_beaten_by_start_does_not_stop_at_epoch_1():
    # regressão: partindo de um best.pt com mAP 0.65 e alvo 0.60, parava na 1ª época por mais épocas que se pedisse
    assert _fit_end(baseline=0.65, map50=0.66) is False
    assert train.STOP['goal'] is False
    assert _fit_end(baseline=0.40, map50=0.61) is True       # parte abaixo do alvo e o atinge: para
    assert train.STOP['goal'] is True
    train.STOP.update(asked=False, goal=False)


def test_epoch_plot_saved_every_epoch(tmp_path):
    hist = [dict(epoch=1, map50=.2, precision=.3, recall=.1), dict(epoch=2, map50=.4, precision=.5, recall=.3)]
    out = train.save_epoch_plot(tmp_path, hist)
    assert (tmp_path / 'epoch_metrics.json').exists()
    assert train.plt is None or (out and out.exists() and out.stat().st_size > 1000)
