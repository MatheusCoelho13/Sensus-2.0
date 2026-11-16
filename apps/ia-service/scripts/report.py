#!/usr/bin/env python3
"""Relatório de avaliação com matriz de confusão (geral e de CADA classe).

  make relatorio                       # usa models/best.pt
  python3 scripts/report.py --weights runs/detect/train_iter/weights/best.pt

Também é gerado automaticamente no fim do treino (train.py). Saída em runs/report/<data-hora>/:
relatorio.html (autocontido), relatorio.json e as imagens do Ultralytics. runs/report/latest.html = o último.

A matriz usa o limiar de confiança --conf (0.25) e IoU 0.5; o mAP vem de outra validação (conf padrão).
Em detecção não existe "verdadeiro negativo", por isso o 2x2 de cada classe mostra só TP/FP/FN.
"""
import argparse
import html
import json
import shutil
import time
from pathlib import Path


# ---------------------------------------------------------------- cálculo (puro, testável)
def per_class(matrix, names):
    """matrix[previsto][real], última linha/coluna = fundo (Ultralytics). Retorna 1 dict por classe."""
    nc = len(matrix) - 1
    out = []
    for c in range(nc):
        tp = matrix[c][c]
        fp = sum(matrix[c]) - tp                                # previu c, era outra coisa ou fundo
        fn = sum(matrix[p][c] for p in range(nc + 1)) - tp      # era c, não foi detectada como c
        missed = matrix[nc][c]                                  # era c e nada foi detectado
        confused = max(((matrix[p][c], p) for p in range(nc) if p != c), default=(0, 0))
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
        out.append(dict(name=names[c], tp=tp, fp=fp, fn=fn, missed=missed, support=tp + fn,
                        precision=prec, recall=rec, f1=f1,
                        top_confusion=(names[confused[1]], confused[0]) if confused[0] > 0 else None))
    return out


def verdict(s):
    """(texto, classe-css). Sempre texto, nunca só cor."""
    if s['support'] == 0:
        return 'sem exemplos na validação', 'dim'
    r, p = s['recall'] or 0, s['precision'] or 0
    if r >= .8 and p >= .8:
        return 'bom', 'reward'
    if r < .5 and p < .5:
        return 'fraco: perde objetos e erra muito', 'punish'
    if r < .5:
        return 'perde muitos objetos', 'punish'
    if p < .5:
        return 'muitos falsos alarmes', 'punish'
    return 'razoável', 'dim'


# ---------------------------------------------------------------- HTML
CSS = """
:root{--bg:#0d0c0a;--ink:#ece6d4;--dim:#8f8977;--rule:#2b2923;--reward:#c8ff2e;--punish:#ff4b2b;
--serif:"Iowan Old Style","Palatino Linotype","URW Palladio L",P052,Georgia,serif;
--mono:ui-monospace,"JetBrains Mono","SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:light){:root{--bg:#f4efe2;--ink:#1c1a14;--dim:#6b6555;--rule:#d8d1bd;--reward:#4f7d00;--punish:#c4310f}}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--ink);font:14px/1.5 var(--mono)}
.wrap{max-width:1180px;margin:0 auto;padding:0 16px 56px}.label{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim)}
header{padding:22px 0 16px;border-bottom:1px solid var(--rule)}header b{letter-spacing:.3em}
h1{font:italic 400 clamp(30px,5vw,52px)/1.05 var(--serif);margin:10px 0 6px}
.meta{color:var(--dim);font-size:12px;word-break:break-all}
.stats{display:flex;flex-wrap:wrap;gap:8px 44px;padding:22px 0}.stats b{display:block;font:italic 400 clamp(40px,7vw,76px)/1 var(--serif)}
section{padding:24px 0;border-top:1px solid var(--rule)}section>.label{margin-bottom:6px}section>p{color:var(--dim);max-width:78ch;margin-bottom:16px}
.scroll{overflow-x:auto}table.cm{border-collapse:separate;border-spacing:2px;margin-top:6px}
.cm th{font-weight:400;color:var(--dim);font-size:12px;padding:2px 6px}
.cm thead th{vertical-align:bottom;height:96px;white-space:nowrap}.cm thead th span{writing-mode:vertical-rl;transform:rotate(180deg)}
.cm tbody th{text-align:right;white-space:nowrap;padding-right:10px}
.cm td{min-width:54px;height:46px;text-align:center;font-variant-numeric:tabular-nums;border-radius:2px;
background:color-mix(in srgb,var(--c) calc(var(--a)*100%),transparent);color:var(--ink)}
.cm td small{display:block;font-size:10px;opacity:.75}.cm td.z{color:var(--rule)}
.axis{font-size:11px;color:var(--dim);margin:2px 0 0 2px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:22px}
.card{border-top:1px solid var(--rule);padding-top:12px}.card h3{font:italic 400 30px/1.1 var(--serif)}
.v{font-size:12px;margin:2px 0 12px}.v.reward{color:var(--reward)}.v.punish{color:var(--punish)}.v.dim{color:var(--dim)}
.m2{display:grid;grid-template-columns:auto 1fr 1fr;gap:3px;font-size:12px;margin-bottom:12px}
.m2 div{padding:7px 8px;border-radius:2px;background:var(--rule)}.m2 .h{background:none;color:var(--dim);font-size:10px;letter-spacing:.08em;text-transform:uppercase;padding:2px}
.m2 .tp{background:color-mix(in srgb,var(--reward) 22%,transparent)}.m2 .bad{background:color-mix(in srgb,var(--punish) 22%,transparent)}
.m2 b{display:block;font:italic 400 24px/1 var(--serif)}
.bar{display:grid;grid-template-columns:78px 1fr 48px;gap:8px;align-items:center;font-size:12px;margin:3px 0}
.bar i{display:block;height:5px;background:var(--rule)}.bar i u{display:block;height:100%;background:var(--ink);text-decoration:none}
.note{margin-top:10px;color:var(--dim);font-size:12px}
footer{border-top:1px solid var(--rule);padding-top:16px;color:var(--dim);font-size:12px;max-width:80ch}
"""


def pct(v):
    return '—' if v is None else f'{100 * v:.0f}%'


def render_html(summary, matrix, names, classes, meta):
    esc, nc = html.escape, len(matrix) - 1
    labels = [esc(str(n)) for n in names]

    # --- heatmap geral: linhas = REAL, colunas = PREVISTO (transposto do Ultralytics)
    head = ''.join(f'<th><span>{n}</span></th>' for n in labels) + '<th><span>nada detectado</span></th>'
    rows = []
    for r in range(nc + 1):
        cells = [matrix[p][r] for p in range(nc + 1)]   # real r, previsto p (p == nc: nada detectado)
        total = sum(cells)
        tds = []
        for p, v in enumerate(cells):
            if r == nc and p == nc:
                tds.append('<td class="z">·</td>')
                continue
            frac = v / total if total else 0
            good = (p == r)
            a = 0 if v == 0 else max(.14, min(1, frac))
            col = 'var(--reward)' if good else 'var(--punish)'
            tds.append(f'<td style="--a:{a:.2f};--c:{col}" title="real {esc(str(names[r]) if r < nc else "fundo")} → previsto '
                       f'{esc(str(names[p]) if p < nc else "nada")}: {int(v)} ({pct(frac)} da linha)">'
                       + (f'{int(v)}<small>{pct(frac)}</small>' if v else '·') + '</td>')
        lab = labels[r] if r < nc else 'fundo (não era objeto)'
        rows.append(f'<tr><th>{lab}</th>{"".join(tds)}</tr>')
    cm = (f'<div class="scroll"><table class="cm"><thead><tr><th></th>{head}</tr></thead>'
          f'<tbody>{"".join(rows)}</tbody></table></div>'
          '<div class="axis">linhas = o que realmente havia · colunas = o que a rede previu</div>')

    # --- cartão por classe
    cards = []
    for s in classes:
        text, cls = verdict(s)
        bars = ''.join(f'<div class="bar"><span>{k}</span><i><u style="width:{100 * (v or 0):.0f}%"></u></i><span>{pct(v)}</span></div>'
                       for k, v in (('precisão', s['precision']), ('recall', s['recall']), ('F1', s['f1']),
                                    ('AP50', s.get('ap50'))) if v is not None or k != 'AP50')
        conf = ''
        if s['missed']:
            conf += f'{int(s["missed"])} de {int(s["support"])} objetos não foram detectados (viraram fundo).<br>'
        if s['top_confusion']:
            conf += f'Mais confundida com <b>{esc(str(s["top_confusion"][0]))}</b> ({int(s["top_confusion"][1])}×).'
        cards.append(
            f'<div class="card"><h3>{esc(str(s["name"]))}</h3><div class="v {cls}">{esc(text)} · {int(s["support"])} objetos reais</div>'
            f'<div class="m2"><div class="h"></div><div class="h">era {esc(str(s["name"]))}</div><div class="h">não era</div>'
            f'<div class="h">previu</div><div class="tp"><b>{int(s["tp"])}</b>acertos</div><div class="bad"><b>{int(s["fp"])}</b>falsos alarmes</div>'
            f'<div class="h">não previu</div><div class="bad"><b>{int(s["fn"])}</b>perdidos</div><div class="h">—</div></div>'
            f'{bars}<div class="note">{conf or "Sem confusões registradas."}</div></div>')

    stats = ''.join(f'<div><span class="label">{k}</span><b>{v}</b></div>' for k, v in (
        ('mAP50', f'{summary["map50"]:.3f}'), ('mAP50-95', f'{summary["map5095"]:.3f}'),
        ('precisão', f'{summary["precision"]:.3f}'), ('recall', f'{summary["recall"]:.3f}')))
    return f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sensus Lab · Relatório</title><style>{CSS}</style></head><body><div class="wrap">
<header><b>SENSUS·LAB</b> <span class="label">relatório de avaliação</span><h1>Onde a rede acerta e onde erra</h1>
<div class="meta">modelo {esc(meta['weights'])} · dados {esc(meta['data'])} · imgsz {meta['imgsz']} · confiança {meta['conf']} · gerado em {esc(meta['when'])}</div></header>
<div class="stats">{stats}</div>
<section><div class="label">matriz de confusão · todas as classes</div>
<p>Cada linha é o que realmente existia; a diagonal (verde) são os acertos e o vermelho são os erros. “Nada detectado” = objetos que a rede perdeu; a linha “fundo” = falsos alarmes.</p>{cm}</section>
<section><div class="label">matriz de confusão · cada classe</div>
<p>Para cada classe: quantos acertos, falsos alarmes e objetos perdidos, com precisão (dos que previu, quantos estavam certos) e recall (dos que existiam, quantos achou).</p>
<div class="grid">{''.join(cards)}</div></section>
<footer>Matriz calculada com confiança ≥ {meta['conf']} e IoU 0.5. Em detecção não há “verdadeiro negativo” (o fundo é infinito), por isso o quadro 2×2 de cada classe mostra só acertos, falsos alarmes e perdidos. mAP e precisão/recall do topo vêm de uma validação separada, com o limiar padrão do Ultralytics.</footer>
</div></body></html>"""


# ---------------------------------------------------------------- execução
def write_report(weights, data, out_root, imgsz=640, batch=16, device='0', conf=0.25, map_metrics=None):
    """Roda as validações e grava o relatório. Retorna o Path do HTML. `map_metrics`: validação já feita (evita repetir)."""
    from ultralytics import YOLO
    out_root = Path(out_root)
    out_dir = out_root / time.strftime('%Y%m%d-%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    common = dict(data=data, imgsz=imgsz, batch=batch, device=device)
    mm = map_metrics or model.val(**common, plots=False)
    cm = model.val(**common, conf=conf, plots=True, project=str(out_dir), name='ultralytics', exist_ok=True)

    names_d = dict(getattr(cm, 'names', None) or mm.names)
    names = [names_d[i] for i in sorted(names_d)]
    matrix = [[float(v) for v in row] for row in cm.confusion_matrix.matrix]
    classes = per_class(matrix, names)
    try:  # AP50 por classe (opcional)
        for i, c in enumerate(mm.box.ap_class_index):
            classes[int(c)]['ap50'] = float(mm.box.class_result(i)[2])
    except Exception:  # noqa: BLE001
        pass
    summary = dict(map50=float(mm.box.map50), map5095=float(mm.box.map), precision=float(mm.box.mp), recall=float(mm.box.mr))
    meta = dict(weights=str(weights), data=str(data), imgsz=imgsz, conf=conf, when=time.strftime('%d/%m/%Y %H:%M'))

    page = render_html(summary, matrix, names, classes, meta)
    (out_dir / 'relatorio.html').write_text(page, encoding='utf-8')
    (out_dir / 'relatorio.json').write_text(json.dumps(
        dict(meta=meta, summary=summary, names=names, matrix_pred_x_real=matrix, classes=classes), indent=2, ensure_ascii=False), encoding='utf-8')
    shutil.copy2(out_dir / 'relatorio.html', out_root / 'latest.html')
    return out_dir / 'relatorio.html'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', default='models/best.pt')
    ap.add_argument('--data', default='config/data.yaml')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--conf', type=float, default=0.25, help='limiar de confiança da matriz')
    ap.add_argument('--out', default='runs/report')
    a = ap.parse_args()
    if not Path(a.weights).exists():
        raise SystemExit(f'❌ pesos não encontrados: {a.weights}')
    device = a.device
    if device == 'auto':
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
    path = write_report(a.weights, a.data, a.out, a.imgsz, a.batch, device, a.conf)
    print(f'📄 Relatório: {path}')


if __name__ == '__main__':
    main()
