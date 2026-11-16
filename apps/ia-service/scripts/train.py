"""scripts/train.py

Treino SUPERVISIONADO do detector (YOLOv8 = rede neural convolucional).

Como a rede aprende (cada batch de cada época):
  1. forward   : imagem -> camadas de neurônios -> caixas + classes previstas
  2. loss      : compara previsão com o rótulo humano (labels YOLO) -> erro
  3. backward  : backpropagation calcula o gradiente do erro p/ cada peso
  4. optimizer : cada peso se ajusta sozinho: w <- w - lr * gradiente
Repete até o erro de validação parar de cair (early stop por `patience`).

Um único `model.train()` (LR com cosine decay + warmup contínuos), em vez de
chunks que reiniciavam o agendamento de LR a cada N épocas.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

DEFAULT_RUN_NAME = 'train_iter'


def parse_args():
    p = argparse.ArgumentParser(description='Treino supervisionado (backprop) do detector')
    p.add_argument('--max-epochs', type=int, default=200)
    p.add_argument('--step', type=int, default=None, help='DEPRECATED: ignorado (treino agora é contínuo)')
    p.add_argument('--target-map', type=float, default=0.60, help='para o treino ao atingir este mAP50')
    p.add_argument('--export-threshold', type=float, default=0.50)
    p.add_argument('--data', default='config/data.yaml')
    p.add_argument('--model', default='yolov8s.pt')
    p.add_argument('--imgsz', type=int, default=768)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='auto')
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--cls-weight', default='', help="'auto' ou lista 'w0,w1,...' (peso de erro por classe)")
    p.add_argument('--patience', type=int, default=30, help='épocas sem melhora antes de parar')
    p.add_argument('--freeze', type=int, default=0, help='congela N primeiras camadas (use 10 com dataset pequeno)')
    p.add_argument('--lr0', type=float, default=None, help='learning rate inicial (default: automático)')
    p.add_argument('--cache', default='ram', choices=['ram', 'disk', 'false'], help='cache das imagens (ram = mais rápido)')
    p.add_argument('--workers', type=int, default=8, help='processos de data loading')
    p.add_argument('--fast-validate', action='store_true', help='validação final em resolução/batch menores')
    p.add_argument('--skip-validate', action='store_true', help='pula validação final de models/best.pt')
    p.add_argument('--run-name', default=DEFAULT_RUN_NAME)
    p.add_argument('--events', default='', help='arquivo .jsonl p/ o painel (scripts/train_ui.py) acompanhar o treino')
    return p.parse_args()


def select_device(arg_device: str, require_cuda: bool = False) -> str:
    if arg_device and arg_device != 'auto':
        return arg_device
    cuda = torch.cuda.is_available()
    if require_cuda and not cuda:
        raise RuntimeError('CUDA requerida, mas não disponível')
    return '0' if cuda else 'cpu'


def find_start_weight(project_root: Path, default: str) -> str:
    """models/best.pt > último runs/detect/*/weights/best.pt > modelo pedido."""
    cands = [p for p in [project_root / 'models' / 'best.pt',
                         *(project_root / 'runs' / 'detect').glob('*/weights/best.pt')] if p.exists()]
    return str(max(cands, key=lambda p: p.stat().st_mtime)) if cands else default


def class_weights_from_data(data_yaml: str):
    """weight[i] = total / (nc * count_i); classe sem anotação -> 1.0. None se não há labels."""
    data = yaml.safe_load(Path(data_yaml).read_text())
    nc = int(data.get('nc', len(data.get('names', []))))
    labels = Path(data.get('path', '.')) / 'labels' / 'train'
    counts = [0] * nc
    for f in labels.glob('*.txt'):
        for ln in f.read_text().splitlines():
            parts = ln.split()
            if parts and parts[0].isdigit() and int(parts[0]) < nc:
                counts[int(parts[0])] += 1
    total = sum(counts)
    if total == 0:
        return None
    return [total / (nc * c) if c else 1.0 for c in counts]


def validate(weight: Path, data: str, imgsz: int, batch: int, device: str, fast: bool) -> float:
    m = YOLO(str(weight)).val(data=data, imgsz=imgsz // 2 if fast else imgsz,
                              batch=max(1, batch // 2) if fast else batch, device=device)
    return float(m.box.map50)


def emit(path: str, **ev):
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({**ev, 't': time.time()}) + '\n')


def first_layer_filters(trainer):
    """Pesos da 1ª camada conv como tiles RGB 0-255 (o painel desenha os "neurônios" mudando)."""
    try:
        w = trainer.model.model[0].conv.weight.detach().float().cpu()
        if w.shape[1] != 3:
            return None
        w = ((w - w.min()) / (w.max() - w.min() + 1e-8) * 255).round().byte()
        return {'k': int(w.shape[-1]), 'tiles': w.permute(0, 2, 3, 1).tolist()}
    except Exception:
        return None


def make_callbacks(target_map: float, cls_weights, events: str = ''):
    """Callbacks que mostram o aprendizado acontecendo + aplicam peso de classe + early-stop."""
    state = {'prev': None}

    def snapshot(trainer):
        return [p.detach().float().cpu().clone() for p in trainer.model.parameters()]

    def on_train_start(trainer):
        state['prev'] = snapshot(trainer)
        emit(events, type='start', epochs=trainer.epochs, device=str(trainer.device), target=target_map,
             filters=first_layer_filters(trainer))
        if cls_weights:
            # Ultralytics não tem arg 'class_weights': injeta no BCE da loss de classificação.
            # (BCEWithLogitsLoss.weight shape (nc,) faz broadcast sobre (batch, anchors, nc).)
            m = trainer.model
            m.criterion = m.init_criterion()
            m.criterion.bce.weight = torch.tensor(cls_weights, device=trainer.device)
            print(f'🎯 Class weights aplicados na loss: {[round(w, 3) for w in cls_weights]}')

    def on_train_epoch_end(trainer):
        # prova de que os pesos mudaram sozinhos via backprop: ||w_depois - w_antes||
        cur = snapshot(trainer)
        delta = sum(((c - p) ** 2).sum().item() for c, p in zip(cur, state['prev'])) ** 0.5
        norm = sum((c ** 2).sum().item() for c in cur) ** 0.5
        state['prev'] = cur
        losses = dict(zip(trainer.loss_names, trainer.tloss.tolist())) if trainer.tloss is not None else {}
        loss = ' '.join(f'{k}={v:.3f}' for k, v in losses.items())
        state.update(loss=losses, delta=delta, delta_pct=100 * delta / norm)
        print(f'🧠 época {trainer.epoch + 1}: loss[{loss}] | Δpesos={delta:.4f} ({100 * delta / norm:.2f}% da norma)')

    def on_fit_epoch_end(trainer):
        map50 = trainer.metrics.get('metrics/mAP50(B)', 0.0)
        # "recompensa/punição" = o erro (loss) guiando o ajuste dos pesos; aprendizado supervisionado,
        # não RL. Aqui só mostramos se a época foi pra frente (acertou mais) ou pra trás (errou mais).
        prev = state.get('map50')
        verdict = None
        if prev is not None and map50 != prev:
            ok = map50 > prev
            verdict = 'reward' if ok else 'punish'
            print(f'{"🏆 RECOMPENSA: acertou mais" if ok else "⚠️ PUNIÇÃO: errou mais"} '
                  f'(mAP50 {prev:.3f} -> {map50:.3f}) — pesos {"reforçados" if ok else "corrigidos pelo gradiente"}')
        state['map50'] = map50
        m = trainer.metrics
        emit(events, type='epoch', epoch=trainer.epoch + 1, epochs=trainer.epochs, verdict=verdict, prev_map50=prev,
             map50=map50, map5095=m.get('metrics/mAP50-95(B)'), precision=m.get('metrics/precision(B)'),
             recall=m.get('metrics/recall(B)'), loss=state.get('loss', {}), delta=state.get('delta'),
             delta_pct=state.get('delta_pct'), filters=first_layer_filters(trainer))
        if map50 >= target_map:
            print(f'✅ mAP50 {map50:.3f} >= {target_map:.3f} — parando')
            emit(events, type='goal', map50=map50)
            trainer.stop = True

    return {'on_train_start': on_train_start, 'on_train_epoch_end': on_train_epoch_end,
            'on_fit_epoch_end': on_fit_epoch_end}


def main():
    args = parse_args()
    root = Path(__file__).parent.parent
    device = select_device(args.device, args.require_cuda)
    print(f'Usando device: {device}')

    cls_weights = None
    if args.cls_weight:
        cls_weights = (class_weights_from_data(args.data) if args.cls_weight.lower() == 'auto'
                       else [float(w) for w in args.cls_weight.split(',')])
        if cls_weights is None:
            print('⚠️ sem anotações para calcular class weights; ignorando')

    start = find_start_weight(root, args.model)
    print(f'🔁 Ponto de partida: {start}')
    model = YOLO(start)
    for event, fn in make_callbacks(args.target_map, cls_weights, args.events).items():
        model.add_callback(event, fn)

    best_pt = root / 'models' / 'best.pt'
    baseline = -1.0
    if best_pt.exists() and not args.skip_validate:
        baseline = validate(best_pt, args.data, args.imgsz, args.batch, device, args.fast_validate)
        print(f'📌 baseline mAP50 = {baseline:.3f}')

    kwargs = dict(data=args.data, epochs=args.max_epochs, imgsz=args.imgsz, batch=args.batch,
                  device=device, name=args.run_name, exist_ok=True, patience=args.patience,
                  cos_lr=True, freeze=args.freeze or None, close_mosaic=10,
                  cache=False if args.cache == 'false' else args.cache, workers=args.workers)
    if args.lr0:
        kwargs['lr0'] = args.lr0
    model.train(**kwargs)

    run_best = Path(model.trainer.best)
    if not run_best.exists():
        print('⚠️ best.pt do run não encontrado — veja runs/detect')
        return
    new = validate(run_best, args.data, args.imgsz, args.batch, device, args.fast_validate)
    print(f'📈 mAP50 do run = {new:.3f} (baseline {baseline:.3f})')
    if new > baseline:
        best_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_best, best_pt)
        print(f'📦 models/best.pt atualizado (mAP50={new:.3f})')


if __name__ == '__main__':
    main()
