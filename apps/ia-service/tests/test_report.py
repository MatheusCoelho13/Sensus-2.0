import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
import report  # noqa: E402

NAMES = ['person', 'chair', 'bottle']
# matrix[previsto][real]; última linha/coluna = fundo. Colunas = REAL.
#            real: person chair bottle fundo
M = [[80, 5, 0, 10],     # previu person
     [4, 30, 2, 6],      # previu chair
     [0, 1, 10, 3],      # previu bottle
     [16, 14, 8, 0]]     # nada detectado (FN) na linha do fundo


def test_per_class_counts():
    p, c, b = report.per_class(M, NAMES)
    assert (p['tp'], p['fp'], p['fn'], p['missed']) == (80, 15, 20, 16)     # fp = 5+0+10 ; fn = 4+0+16
    assert p['support'] == 100 and abs(p['recall'] - .8) < 1e-9 and abs(p['precision'] - 80 / 95) < 1e-9
    assert (c['tp'], c['fp'], c['fn']) == (30, 12, 20)
    assert c['top_confusion'] == ('person', 5)                               # chair real previsto como person
    assert b['top_confusion'] == ('chair', 2)


def test_verdict_uses_words():
    ok = dict(support=10, precision=.9, recall=.9)
    assert report.verdict(ok) == ('bom', 'reward')
    assert report.verdict(dict(support=10, precision=.9, recall=.2))[1] == 'punish'
    assert report.verdict(dict(support=0, precision=None, recall=None))[0] == 'sem exemplos na validação'


def test_render_escapes_and_lists_every_class():
    names = ['<b>x</b>', 'chair', 'bottle']
    page = report.render_html(dict(map50=.5, map5095=.3, precision=.6, recall=.7), M, names,
                              report.per_class(M, names), dict(weights='w.pt', data='d.yaml', imgsz=640, conf=.25, when='hoje'))
    assert '<b>x</b>' not in page and '&lt;b&gt;x&lt;/b&gt;' in page
    assert page.count('class="card"') == 3 and 'nada detectado' in page and 'fundo' in page


def test_write_report_end_to_end(tmp_path):
    import json
    import types

    class FakeYOLO:
        def __init__(self, w): pass

        def val(self, **kw):
            box = types.SimpleNamespace(map50=.61, map=.38, mp=.7, mr=.6, ap_class_index=[0, 1, 2],
                                        class_result=lambda i: (.7, .6, .5 + i / 10, .3))
            return types.SimpleNamespace(box=box, names={0: 'person', 1: 'chair', 2: 'bottle'},
                                         confusion_matrix=types.SimpleNamespace(matrix=M))

    mod = types.ModuleType('ultralytics')
    mod.YOLO = FakeYOLO
    old = sys.modules.get('ultralytics')
    sys.modules['ultralytics'] = mod
    try:
        out = report.write_report('best.pt', 'data.yaml', tmp_path, device='cpu')
    finally:
        sys.modules['ultralytics'] = old if old else types.ModuleType('ultralytics')
    assert out.name == 'relatorio.html' and (tmp_path / 'latest.html').read_text() == out.read_text()
    data = json.loads((out.parent / 'relatorio.json').read_text())
    assert data['names'] == NAMES and data['classes'][0]['tp'] == 80 and data['classes'][2]['ap50'] == .7
    assert data['summary']['map50'] == .61
