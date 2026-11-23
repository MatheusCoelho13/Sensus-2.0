#!/usr/bin/env python3
"""scripts/improve_dataset.py

Melhora o dataset YOLO (padrão: config/data.yaml -> datasets/coco_nav) em 3 etapas:

  1. AUDITA   : imagens ilegíveis, rótulos órfãos, linhas inválidas, caixas minúsculas/duplicadas,
                imagens duplicadas (md5) e VAZAMENTO train/val (mesma imagem nos dois lados).
  2. CORRIGE  : (--fix) conserta linhas (clip das caixas, remove inválidas) e move o que não tem
                conserto p/ datasets/<nome>/_quarantine/ (nada é apagado; dá p/ devolver).
  3. BALANCEIA: (--balance) gera variações (flip, rotação, brilho/contraste, desfoque, ruído, cor) das
                imagens que contêm as classes com poucos exemplos, até chegar a --target instâncias.
                Só no `train`; o `val` nunca é tocado.

Sem flags só audita e imprime o relatório (não altera nada).

  python scripts/improve_dataset.py                          # auditoria
  python scripts/improve_dataset.py --fix                    # + correções
  python scripts/improve_dataset.py --fix --balance          # + balanceia até a mediana das classes
  python scripts/improve_dataset.py --fix --balance --target 1500
  python scripts/improve_dataset.py --clean                  # remove só o que o balanceamento gerou
"""
import argparse
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXT = {'.jpg', '.jpeg', '.png'}
TAG = re.compile(r'_aug[a-z0-9]+$|_rot[mp]\d+$')   # nomes gerados por este script e por augment_rotate.py
MIN_SIDE = 0.005          # caixa com lado < 0,5% da imagem é ruído de anotação
MIN_VISIBLE = 0.4         # rotação: descarta caixa com <40% da área dentro da imagem
SEED = 1234


# ───────────────────────── leitura ─────────────────────────
def load_cfg(path):
    cfg = yaml.safe_load(Path(path).read_text())
    names = cfg['names'] if isinstance(cfg['names'], list) else [cfg['names'][k] for k in sorted(cfg['names'])]
    base = Path(cfg.get('path', '.'))
    d = {s: (base / cfg[s], base / str(cfg[s]).replace('images', 'labels', 1)) for s in ('train', 'val') if cfg.get(s)}
    return base, names, d


def images(d: Path):
    return sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT and not TAG.search(f.stem)) if d.is_dir() else []


def parse_label(path: Path):
    """-> (linhas válidas [(c,cx,cy,w,h)], nº de linhas inválidas descartadas, nº de caixas corrigidas por clip)."""
    rows, bad, fixed = [], 0, 0
    for ln in path.read_text().splitlines():
        p = ln.split()
        if not p:
            continue
        try:
            if len(p) != 5:
                raise ValueError
            c, *v = int(p[0]), *map(float, p[1:])
        except ValueError:
            bad += 1
            continue
        cx, cy, w, h = v
        x1, y1, x2, y2 = max(cx - w / 2, 0), max(cy - h / 2, 0), min(cx + w / 2, 1), min(cy + h / 2, 1)
        if (x1, y1, x2, y2) != (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2):
            fixed += 1
        w, h = x2 - x1, y2 - y1
        if c < 0 or w < MIN_SIDE or h < MIN_SIDE:
            bad += 1
            continue
        rows.append((c, (x1 + x2) / 2, (y1 + y2) / 2, w, h))
    return rows, bad, fixed


def write_label(path: Path, rows):
    path.write_text(''.join(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n' for c, x, y, w, h in rows))


def md5(p: Path):
    return hashlib.md5(p.read_bytes()).hexdigest()


# ───────────────────────── 1+2: audita e corrige ─────────────────────────
def audit(base, names, splits, fix):
    nc, q = len(names), base / '_quarantine'
    rep = Counter()
    seen = {}            # md5 -> (split, path)
    val_hash = {md5(i) for i in images(splits['val'][0])} if 'val' in splits else set()

    def quarantine(*files, why):
        rep[why] += 1
        if fix:
            for f in files:
                if f and f.exists():
                    dest = q / why / f.parent.parent.name / f.parent.name
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), dest / f.name)

    for split in ('train', 'val'):
        if split not in splits:
            continue
        idir, ldir = splits[split]
        for img in images(idir):
            lab = ldir / f'{img.stem}.txt'
            if cv2.imread(str(img)) is None:
                quarantine(img, lab, why='imagem_ilegivel')
                continue
            h = md5(img)
            if split == 'train' and h in val_hash:
                quarantine(img, lab, why='vazamento_train_val')   # avaliar em imagem vista no treino infla o mAP
                continue
            if h in seen:
                quarantine(img, lab, why=f'duplicada_{split}')
                continue
            seen[h] = (split, img)
            if not lab.exists():
                rep['sem_rotulo(fundo)'] += 1
                continue
            rows, bad, fixed = parse_label(lab)
            rows = list(dict.fromkeys(rows))   # linhas repetidas
            if any(c >= nc for c, *_ in rows):
                bad += sum(c >= nc for c, *_ in rows)
                rows = [r for r in rows if r[0] < nc]
            if bad or fixed or len(rows) != len(lab.read_text().split()) // 5:
                rep['linhas_invalidas'] += bad
                rep['caixas_ajustadas_clip'] += fixed
                if fix:
                    write_label(lab, rows)
        for lab in ldir.glob('*.txt') if ldir.is_dir() else []:
            if not TAG.search(lab.stem) and not any((idir / f'{lab.stem}{e}').exists() for e in IMG_EXT):
                quarantine(lab, why='rotulo_orfao')
    return rep


# ───────────────────────── contagem ─────────────────────────
def count_instances(ldir: Path, nc, include_generated=False):
    inst, imgs = Counter(), defaultdict(list)
    for lab in sorted(ldir.glob('*.txt')) if ldir.is_dir() else []:
        if not include_generated and TAG.search(lab.stem):
            continue
        rows, _, _ = parse_label(lab)
        for c, *_ in rows:
            if c < nc:
                inst[c] += 1
                imgs[c].append((lab, len(rows)))
    return inst, imgs


# ───────────────────────── 3: balanceamento ─────────────────────────
def rot_matrix(w, h, ang):
    return cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)


def rotate_rows(rows, M, w, h):
    out = []
    for c, cx, cy, bw, bh in rows:
        x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]]) @ M.T
        nx1, ny1, nx2, ny2 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
        a = (nx2 - nx1) * (ny2 - ny1)
        x1, y1, x2, y2 = max(nx1, 0), max(ny1, 0), min(nx2, w), min(ny2, h)
        if x2 <= x1 or y2 <= y1 or a <= 0 or (x2 - x1) * (y2 - y1) / a < MIN_VISIBLE or (x2 - x1) / w < MIN_SIDE or (y2 - y1) / h < MIN_SIDE:
            continue
        out.append((c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return out


def t_flip(img, rows, rng):
    return cv2.flip(img, 1), [(c, 1 - x, y, w, h) for c, x, y, w, h in rows]


def t_rot(ang):
    def f(img, rows, rng):
        h, w = img.shape[:2]
        M = rot_matrix(w, h, ang)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114)), rotate_rows(rows, M, w, h)
    return f


def t_light(img, rows, rng):   # brilho/contraste (luz ruim dentro de casa, contraluz)
    a, b = rng.uniform(0.6, 1.4), rng.uniform(-35, 35)
    return cv2.convertScaleAbs(img, alpha=a, beta=b), rows


def t_blur(img, rows, rng):    # movimento/foco ruim de câmera de celular
    k = int(rng.choice([3, 5, 7]))
    return cv2.GaussianBlur(img, (k, k), 0), rows


def t_noise(img, rows, rng):   # sensor com pouca luz
    n = rng.normal(0, rng.uniform(5, 15), img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8), rows


def t_hsv(img, rows, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-10, 10)) % 180
    hsv[..., 1] *= rng.uniform(0.6, 1.4)
    hsv[..., 2] *= rng.uniform(0.7, 1.3)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR), rows


def chain(*fs):
    def f(img, rows, rng):
        for g in fs:
            img, rows = g(img, rows, rng)
        return img, rows
    return f


TRANSFORMS = [('flip', t_flip), ('light', t_light), ('rotm8', t_rot(-8)), ('rotp8', t_rot(8)), ('hsv', t_hsv),
              ('blur', t_blur), ('rotm15', t_rot(-15)), ('rotp15', t_rot(15)), ('noise', t_noise),
              ('fliplight', chain(t_flip, t_light)), ('flipp8', chain(t_flip, t_rot(8))), ('flipm8', chain(t_flip, t_rot(-8)))]


def balance(names, ldir, idir, target, max_per_source):
    nc = len(names)
    inst, srcs = count_instances(ldir, nc)
    if not inst:
        print('⚠️ sem anotações p/ balancear')
        return 0
    goal = target or int(np.median(list(inst.values())))
    print(f'\n⚖️  Balanceando o train até ~{goal} instâncias por classe (mediana/alvo)')
    rng, made = np.random.default_rng(SEED), 0
    gained = Counter()
    for c in sorted(inst, key=inst.get):   # as mais raras primeiro
        need = goal - inst[c] - gained[c]
        if need <= 0:
            continue
        # fontes com poucas caixas primeiro: variação mais "limpa" da classe alvo e menos inflação de `person`
        pool = sorted(srcs[c], key=lambda t: t[1])
        for rnd in range(max_per_source):
            progress = False
            for lab, _ in pool:
                if need <= 0:
                    break
                tag, fn = TRANSFORMS[(rnd + sum(map(ord, lab.stem))) % len(TRANSFORMS)]
                out_stem = f'{lab.stem}_aug{tag}'
                out_lab = ldir / f'{out_stem}.txt'
                img_path = next((idir / f'{lab.stem}{e}' for e in IMG_EXT if (idir / f'{lab.stem}{e}').exists()), None)
                if out_lab.exists() or img_path is None:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                rows, _, _ = parse_label(lab)
                new_img, new_rows = fn(img, rows, rng)
                got = sum(1 for r in new_rows if r[0] == c)
                if not got:
                    continue
                cv2.imwrite(str(idir / f'{out_stem}.jpg'), new_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                write_label(out_lab, new_rows)
                for r in new_rows:
                    gained[r[0]] += 1
                need -= got
                made += 1
                progress = True
            if need <= 0 or not progress:
                break
        print(f'   {names[c]:<12} {inst[c]:>6} → {inst[c] + gained[c]:>6}' + ('' if need <= 0 else f'  (faltou {need}: poucas imagens-fonte; baixe mais com enrich_dataset.py)'))
    return made


def clean_generated(splits):
    n = 0
    idir, ldir = splits['train']
    for d in (idir, ldir):
        for f in d.iterdir():
            if TAG.search(f.stem):
                f.unlink()
                n += 1
    (ldir.parent / 'train.cache').unlink(missing_ok=True)
    print(f'🧹 {n} arquivos gerados removidos')


def table(names, splits):
    print(f'\n{"classe":<12} {"train":>7} {"val":>7}   (instâncias originais)')
    tr, _ = count_instances(splits['train'][1], len(names))
    va, _ = count_instances(splits['val'][1], len(names)) if 'val' in splits else (Counter(), None)
    top = max(tr.values(), default=1)
    for i, n in enumerate(names):
        print(f'{n:<12} {tr[i]:>7} {va[i]:>7}   {"█" * int(30 * tr[i] / top)}')
    if tr and max(tr.values()) / max(min(tr.values()), 1) > 3:
        print(f'⚠️ desbalanceado: a maior classe tem {max(tr.values()) / max(min(tr.values()), 1):.1f}× a menor — use --balance')


def main():
    ap = argparse.ArgumentParser(description='Audita, corrige e balanceia o dataset YOLO')
    ap.add_argument('--data', default='config/data.yaml')
    ap.add_argument('--fix', action='store_true', help='aplica correções (move o irrecuperável p/ _quarantine/, não apaga)')
    ap.add_argument('--balance', action='store_true', help='gera variações das classes raras (só no train)')
    ap.add_argument('--target', type=int, default=0, help='instâncias por classe a atingir (0 = mediana)')
    ap.add_argument('--max-per-source', type=int, default=4, help='máx. de variações por imagem-fonte')
    ap.add_argument('--clean', action='store_true', help='remove as imagens geradas (_aug*/_rot*)')
    a = ap.parse_args()

    base, names, splits = load_cfg(a.data)
    if 'train' not in splits or not splits['train'][0].is_dir():
        raise SystemExit(f'pasta de treino não encontrada em {base}')
    if a.clean:
        return clean_generated(splits)

    print(f'📂 {base} · {len(names)} classes · {"CORRIGINDO" if a.fix else "só auditoria (use --fix p/ aplicar)"}')
    rep = audit(base, names, splits, a.fix)
    print('\n🔎 Auditoria' + (' (aplicada)' if a.fix else ' (nada alterado)'))
    for k, v in sorted(rep.items()):
        print(f'   {k:<24} {v}')
    if not rep:
        print('   nenhum problema encontrado ✅')
    table(names, splits)

    if a.balance:
        made = balance(names, splits['train'][1], splits['train'][0], a.target, a.max_per_source)
        (splits['train'][1].parent / 'train.cache').unlink(missing_ok=True)
        print(f'\n✅ {made} imagens novas no train. Desfazer: python scripts/improve_dataset.py --clean')
    elif a.fix:
        (splits['train'][1].parent / 'train.cache').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
