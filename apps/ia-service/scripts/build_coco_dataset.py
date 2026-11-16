#!/usr/bin/env python3
"""Baixa o COCO 2017 só com as classes do projeto, BALANCEIA por classe e exporta em YOLO.

  make dataset                      # 1000 imgs/classe no train, 200 no val
  make dataset PER_CLASS=2000

Passos: anotações oficiais (241MB) -> seleciona imagens balanceadas -> baixa só
essas imagens (coco_url) -> escreve labels YOLO -> gera config/data.yaml.
Train vem do train2017 e val do val2017 oficiais: sem vazamento entre os dois.
Retomável: imagens já baixadas são puladas.
"""
import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CLASSES = ['person', 'backpack', 'chair', 'bench', 'laptop', 'cell phone', 'bottle', 'book', 'cup', 'tv']
ANN_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'


def select_balanced(img_classes: dict, per_class: int, seed: int = 0) -> set:
    """img_classes: image_id -> lista de class ids. Devolve ids escolhidos.

    Classes mais raras primeiro; cada imagem escolhida conta p/ TODAS as classes
    que contém, então a classe comum (person) enche sozinha e para de entrar.
    """
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for iid, cs in img_classes.items():
        for c in set(cs):
            by_cls[c].append(iid)
    chosen, count = set(), Counter()
    for c in sorted(by_cls, key=lambda k: (len(by_cls[k]), k)):
        pool = sorted(by_cls[c])
        rng.shuffle(pool)
        for iid in pool:
            if count[c] >= per_class:
                break
            if iid in chosen:
                continue
            chosen.add(iid)
            count.update(set(img_classes[iid]))
    return chosen


def yolo_lines(img: dict, anns: list) -> list:
    W, H = img['width'], img['height']
    out = []
    for cls, (x, y, w, h) in anns:
        if w < 2 or h < 2:
            continue
        cx, cy = min(max((x + w / 2) / W, 0), 1), min(max((y + h / 2) / H, 0), 1)
        out.append(f'{cls} {cx:.6f} {cy:.6f} {min(w / W, 1):.6f} {min(h / H, 1):.6f}')
    return out


def fetch(url: str, dst: Path, retries: int = 3) -> bool:
    import requests
    if dst.exists():
        return True
    tmp = dst.with_suffix(dst.suffix + '.part')
    for _ in range(retries):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            tmp.rename(dst)
            return True
        except Exception:
            tmp.unlink(missing_ok=True)
    return False


def get_annotations_zip(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    z = cache / 'annotations_trainval2017.zip'
    if not z.exists():
        print('⬇️  Baixando anotações COCO (241MB)...')
        if not fetch(ANN_URL, z):
            raise SystemExit('❌ falha ao baixar anotações')
    return z


def load_split(zf: zipfile.ZipFile, split: str):
    with zf.open(f'annotations/instances_{split}2017.json') as f:
        d = json.load(f)
    cid = {c['id']: CLASSES.index(c['name']) for c in d['categories'] if c['name'] in CLASSES}
    imgs = {i['id']: i for i in d['images']}
    anns = defaultdict(list)
    for a in d['annotations']:
        if not a['iscrowd'] and a['category_id'] in cid:
            anns[a['image_id']].append((cid[a['category_id']], a['bbox']))
    return imgs, anns


def build_split(zf, split: str, name: str, per_class: int, out: Path, seed: int, workers: int):
    imgs, anns = load_split(zf, split)
    chosen = select_balanced({i: [c for c, _ in a] for i, a in anns.items()}, per_class, seed)
    img_dir, lbl_dir = out / 'images' / name, out / 'labels' / name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for d in (img_dir, lbl_dir):  # remove sobras de execuções com outros parâmetros
        for f in d.iterdir():
            if f.stem not in {Path(imgs[i]['file_name']).stem for i in chosen}:
                f.unlink()

    def job(iid):
        im = imgs[iid]
        if not fetch(im['coco_url'], img_dir / im['file_name']):
            return None
        (lbl_dir / (Path(im['file_name']).stem + '.txt')).write_text('\n'.join(yolo_lines(im, anns[iid])))
        return iid

    with ThreadPoolExecutor(workers) as ex:
        try:
            from tqdm import tqdm
            done = [r for r in tqdm(ex.map(job, sorted(chosen)), total=len(chosen), desc=f'{name}') if r]
        except ImportError:
            done = [r for r in ex.map(job, sorted(chosen)) if r]
    img_count, inst_count = Counter(), Counter()
    for iid in done:
        for c in {c for c, _ in anns[iid]}:
            img_count[c] += 1
        inst_count.update(c for c, _ in anns[iid])
    print(f'\n📊 {name}: {len(done)}/{len(chosen)} imagens')
    for i, n in enumerate(CLASSES):
        print(f'   {n:<11} imagens={img_count[i]:>5}  instâncias={inst_count[i]:>6}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='datasets/coco_nav')
    ap.add_argument('--per-class', type=int, default=1000, help='imagens por classe no train')
    ap.add_argument('--val-per-class', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--cache-dir', default='data/coco_raw')
    ap.add_argument('--yaml', default='config/data.yaml')
    a = ap.parse_args()

    out = Path(a.out).resolve()
    with zipfile.ZipFile(get_annotations_zip(Path(a.cache_dir))) as zf:
        build_split(zf, 'train', 'train', a.per_class, out, a.seed, a.workers)
        build_split(zf, 'val', 'val', a.val_per_class, out, a.seed, a.workers)

    import yaml
    Path(a.yaml).write_text(yaml.safe_dump(
        {'path': str(out), 'train': 'images/train', 'val': 'images/val', 'nc': len(CLASSES), 'names': CLASSES},
        sort_keys=False, allow_unicode=True))
    print(f'\n✅ Dataset em {out}\n📝 {a.yaml} atualizado\n▶  make train_coco')


if __name__ == '__main__':
    main()
