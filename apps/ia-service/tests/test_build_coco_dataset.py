import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from build_coco_dataset import select_balanced, yolo_lines  # noqa: E402


def test_balance_caps_common_class_and_keeps_rare():
    # 500 imagens só com classe 0 (comum), 10 só com 1, 5 só com 2 (raras)
    data = {i: [0] for i in range(500)}
    data.update({1000 + i: [1] for i in range(10)})
    data.update({2000 + i: [2] for i in range(5)})
    chosen = select_balanced(data, per_class=20, seed=1)
    cnt = Counter(c for i in chosen for c in set(data[i]))
    assert cnt[0] == 20 and cnt[1] == 10 and cnt[2] == 5


def test_rare_class_fills_common_class():
    # toda imagem rara também tem person: person não precisa de imagens extras
    data = {i: [1, 0] for i in range(30)}
    data.update({100 + i: [0] for i in range(500)})
    chosen = select_balanced(data, per_class=30, seed=0)
    assert len(chosen) == 30


def test_yolo_lines_normalizes_and_drops_tiny():
    img = {'width': 100, 'height': 200}
    out = yolo_lines(img, [(3, (10, 20, 40, 100)), (1, (0, 0, 1, 1))])
    assert out == ['3 0.300000 0.350000 0.400000 0.500000']
