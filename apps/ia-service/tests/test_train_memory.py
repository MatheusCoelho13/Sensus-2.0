import sys
import tempfile
import types
from pathlib import Path

# torch/ultralytics não são necessários p/ testar a lógica de memória
for name in ('torch', 'ultralytics'):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules['ultralytics'].YOLO = object
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
import train  # noqa: E402


def make_dataset(n_train=100, n_val=20):
    root = Path(tempfile.mkdtemp())
    for split, n in (('train', n_train), ('val', n_val)):
        d = root / 'images' / split
        d.mkdir(parents=True)
        for i in range(n):
            (d / f'{i}.jpg').write_bytes(b'x')
    y = root / 'data.yaml'
    y.write_text(f'path: {root}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [a]\n')
    return str(y)


def test_estimate_counts_train_and_val():
    est = train.estimate_cache_gb(make_dataset(), 640)
    assert abs(est - 120 * 640 * 640 * 0.75 * 3 / 1e9) < 1e-9


def test_cache_ram_only_when_it_fits():
    y = make_dataset()                         # ~0.11 GB estimados
    assert train.pick_cache('ram', 1.0, y, 640) == 'ram'
    assert train.pick_cache('ram', 0.05, y, 640) is False   # estoura o limite -> desliga
    assert train.pick_cache('disk', 0.05, y, 640) == 'disk'  # limite só vale p/ cache em RAM
    assert train.pick_cache('false', 1.0, y, 640) is False


def test_auto_limit_uses_free_ram():
    y = make_dataset()
    train.available_ram_gb = lambda: 0.1       # 60% de 0.1 = 0.06GB < 0.11GB
    assert train.pick_cache('ram', 0, y, 640) is False
    train.available_ram_gb = lambda: 100
    assert train.pick_cache('ram', 0, y, 640) == 'ram'


def _fake_gpu(total_gb=4.0, available=True):
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=total_gb * 1e9))
    sys.modules['torch'].cuda = cuda


def test_batch_zero_means_max_gpu():
    _fake_gpu(4.0)
    assert train.resolve_batch(16, '0') == (16, 16)              # fixo continua fixo
    assert train.resolve_batch(0, '0') == (0.9, 16)              # máximo: 90% da GPU inteira
    assert train.resolve_batch(0, '0', vram_gb=2) == (0.45, 16)  # máximo dentro do limite de VRAM
    assert train.resolve_batch(0, '0', vram_gb=99) == (0.9, 16)  # limite maior que a GPU = GPU inteira
    assert train.resolve_batch(0, 'cpu') == (16, 16)
    _fake_gpu(available=False)
    assert train.resolve_batch(0, '0') == (16, 16)               # sem CUDA


def test_workers_zero_means_all_cores():
    import os
    assert train.resolve_workers(0) == (os.cpu_count() or 8)
    assert train.resolve_workers(3) == 3


def test_ram_zero_means_all_free_ram():
    y = make_dataset()
    train.available_ram_gb = lambda: 0.2       # 0 = toda a RAM livre (0.2GB) >= ~0.11GB estimados
    assert train.pick_cache('ram', 0, y, 640) == 'ram'
    train.available_ram_gb = lambda: 0.05
    assert train.pick_cache('ram', 0, y, 640) is False
