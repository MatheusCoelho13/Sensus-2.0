import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
import train_ui as core  # noqa: E402


def test_images_listed_from_run_dir_and_path_traversal_blocked(tmp_path):
    (tmp_path / 'results.png').write_bytes(b'x')
    (tmp_path / 'epoch_metrics.png').write_bytes(b'x')
    (tmp_path / 'segredo.png').write_bytes(b'x')                       # fora da lista: nunca é servido
    ev = [{'type': 'start', 'run_dir': str(tmp_path)}]
    names = [i['name'] for i in core.list_images(ev)]
    assert names == ['epoch_metrics.png', 'results.png']               # ordem da lista, só as existentes
    assert all(i['caption'] for i in core.list_images(ev))
    assert core.image_path('results.png', ev) == tmp_path / 'results.png'
    assert core.image_path('segredo.png', ev) is None
    assert core.image_path('../results.png', ev) is None
    assert core.image_path('../../etc/passwd', ev) is None
    assert core.image_path('labels.jpg', ev) is None                   # na lista, mas não existe


def test_run_dir_falls_back_to_default():
    assert core.run_dir([]).name == 'train_iter'
    assert core.run_dir([{'type': 'epoch'}]).name == 'train_iter'


def test_new_ultralytics_box_prefixed_curves_shown_once(tmp_path):
    # o Ultralytics novo grava BoxPR_curve.png etc.; o antigo PR_curve.png. Mostra a que existir, sem duplicar.
    for n in ('BoxPR_curve.png', 'PR_curve.png', 'BoxF1_curve.png', 'train_batch2.jpg', 'results.png'):
        (tmp_path / n).write_bytes(b'x')
    (tmp_path / 'weights').mkdir()
    (tmp_path / 'weights' / 'best.pt').write_bytes(b'x' * 10)
    (tmp_path / 'weights' / 'last.pt').write_bytes(b'x' * 10)
    (tmp_path / 'results.csv').write_text('epoch\n')
    ev = [{'type': 'start', 'run_dir': str(tmp_path)}]
    names = [i['name'] for i in core.list_images(ev)]
    assert names == ['results.png', 'BoxPR_curve.png', 'BoxF1_curve.png', 'train_batch2.jpg']   # PR aparece 1x (a Box)
    assert [f['name'] for f in core.list_files(ev)] == ['weights/best.pt', 'weights/last.pt', 'results.csv']
    assert core.image_path('BoxPR_curve.png', ev) and core.image_path('BoxR_curve.png', ev) is None
