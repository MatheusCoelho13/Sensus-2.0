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
import signal
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

DEFAULT_RUN_NAME = 'train_iter'
STOP = {'asked': False}


def _ask_stop(signum, frame):
    """q / Ctrl+C: termina a época atual e salva best.pt/last.pt (2ª vez = aborta na hora)."""
    if STOP['asked']:
        raise KeyboardInterrupt
    STOP['asked'] = True
    print('🛑 parada pedida: terminando a época e salvando best.pt/last.pt… (de novo = abortar)')


def parse_args():
    p = argparse.ArgumentParser(description='Treino supervisionado (backprop) do detector')
    p.add_argument('--max-epochs', type=int, default=200)
    p.add_argument('--step', type=int, default=None, help='DEPRECATED: ignorado (treino agora é contínuo)')
    p.add_argument('--target-map', type=float, default=0.60, help='para o treino ao atingir este mAP50')
    p.add_argument('--export-threshold', type=float, default=0.50)
    p.add_argument('--data', default='config/data.yaml')
    p.add_argument('--model', default=None, help='peso .pt inicial (ex: models/meu.pt). Sem ele: continua do models/best.pt ou usa yolov8s.pt')
    p.add_argument('--imgsz', type=int, default=768)
    p.add_argument('--batch', type=int, default=0, help='imagens por passo (0 = máximo: ocupa 90%% da VRAM; em CPU usa 16)')
    p.add_argument('--device', default='auto')
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--cls-weight', default='', help="'auto' ou lista 'w0,w1,...' (peso de erro por classe)")
    p.add_argument('--patience', type=int, default=30, help='épocas sem melhora antes de parar')
    p.add_argument('--freeze', type=int, default=0, help='congela N primeiras camadas (use 10 com dataset pequeno)')
    p.add_argument('--lr0', type=float, default=None, help='learning rate inicial (default: automático)')
    p.add_argument('--cache', default='ram', choices=['ram', 'disk', 'false'], help='cache das imagens (ram = mais rápido)')
    p.add_argument('--workers', type=int, default=0, help='processos de data loading (0 = máximo: todos os núcleos da CPU)')
    p.add_argument('--ram-gb', type=float, default=0, help='limite de RAM p/ o cache de imagens em GB (0 = máximo: toda a RAM livre)')
    p.add_argument('--vram-gb', type=float, default=0, help='limite de VRAM do processo em GB (0 = máximo: a GPU inteira); passou do limite dá OOM em vez de travar a máquina')
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


def available_ram_gb():
    """RAM livre (GB) via /proc/meminfo; None fora do Linux."""
    try:
        for ln in Path('/proc/meminfo').read_text().splitlines():
            if ln.startswith('MemAvailable:'):
                return int(ln.split()[1]) / 1e6
    except OSError:
        pass
    return None


def estimate_cache_gb(data_yaml: str, imgsz: int) -> float:
    """Imagens decodificadas (maior lado = imgsz, ~4:3, 3 canais) de train+val."""
    d = yaml.safe_load(Path(data_yaml).read_text())
    base, n = Path(d.get('path', '.')), 0
    for split in ('train', 'val'):
        rel = d.get(split)
        if rel and (base / rel).is_dir():
            n += sum(1 for f in (base / rel).rglob('*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    return n * imgsz * imgsz * 0.75 * 3 / 1e9


def pick_cache(cache: str, ram_gb: float, data_yaml: str, imgsz: int) -> str:
    """cache=ram só se couber no limite (--ram-gb; 0 = toda a RAM livre); senão desliga em vez de estourar a RAM."""
    if cache != 'ram':
        return False if cache == 'false' else cache
    limit = ram_gb or available_ram_gb() or None
    if limit is None:
        return cache
    est = estimate_cache_gb(data_yaml, imgsz)
    if est <= limit:
        print(f'🧠 cache em RAM: ~{est:.1f}GB de {limit:.1f}GB permitidos')
        return 'ram'
    print(f'⚠️ cache em RAM (~{est:.1f}GB) passa do limite de {limit:.1f}GB: cache desligado (aumente --ram-gb ou use --cache disk)')
    return False


def cuda_index(device) -> int:
    last = str(device).split(':')[-1]
    return int(last) if last.isdigit() else 0


def has_gpu(device) -> bool:
    return str(device) != 'cpu' and torch.cuda.is_available()


def resolve_batch(batch: int, device, vram_gb: float = 0):
    """(batch de treino, batch de validação). batch>0 é fixo. 0 = máximo: o Ultralytics aceita fração 0-1
    e dimensiona o batch p/ ocupar essa parte da VRAM (90%% da GPU, ou do limite --vram-gb). Em CPU: 16."""
    if batch > 0:
        return batch, batch
    if not has_gpu(device):
        return 16, 16
    total = torch.cuda.get_device_properties(cuda_index(device)).total_memory / 1e9
    frac = 0.9 * (min(1.0, vram_gb / total) if vram_gb else 1.0)
    print(f'🚀 batch máximo: ocupar {frac:.0%} dos {total:.1f}GB da GPU')
    return round(frac, 2), 16


def resolve_workers(workers: int) -> int:
    import os
    return workers or os.cpu_count() or 8


def apply_vram_limit(device: str, gb: float):
    if not gb or not has_gpu(device):
        return
    idx = cuda_index(device)
    total = torch.cuda.get_device_properties(idx).total_memory / 1e9
    frac = min(1.0, gb / total)
    torch.cuda.set_per_process_memory_fraction(frac, idx)
    print(f'🔒 VRAM limitada a {gb:.1f}GB ({frac:.0%} de {total:.1f}GB)')


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


def val_metrics(weight: Path, data: str, imgsz: int, batch: int, device: str, fast: bool):
    return YOLO(str(weight)).val(data=data, imgsz=imgsz // 2 if fast else imgsz,
                                 batch=max(1, batch // 2) if fast else batch, device=device)


def validate(weight: Path, data: str, imgsz: int, batch: int, device: str, fast: bool) -> float:
    return float(val_metrics(weight, data, imgsz, batch, device, fast).box.map50)


def emit(path: str, **ev):
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({**ev, 't': time.time()}) + '\n')


def first_layer_filters(trainer, prev=None):
    """(payload, pesos_brutos). Neurônios da 1ª camada conv: tiles RGB 0-255 normalizados POR filtro
    (mostra o padrão de cada um) + delta = quanto cada filtro mudou desde `prev`. O painel desenha isso."""
    try:
        w = trainer.model.model[0].conv.weight.detach().float().cpu()
        if w.shape[1] != 3:
            return None, None
        flat = w.flatten(1)
        lo, hi = flat.min(1, keepdim=True).values, flat.max(1, keepdim=True).values
        tiles = ((flat - lo) / (hi - lo + 1e-8) * 255).round().byte().view_as(w)
        delta = (w - prev).flatten(1).norm(dim=1).tolist() if prev is not None and prev.shape == w.shape else None
        return {'k': int(w.shape[-1]), 'tiles': tiles.permute(0, 2, 3, 1).tolist(), 'delta': delta}, w
    except Exception:
        return None, None


def epoch_losses(trainer) -> dict:
    """Loss média da época. Ultralytics >= 8.4: tloss é dict {nome: valor}; antes era tensor + loss_names."""
    t = getattr(trainer, 'tloss', None)
    if t is None:
        return {}
    if isinstance(t, dict):
        return {k: float(v) for k, v in t.items()}
    names = getattr(trainer, 'loss_names', None) or ('box_loss', 'cls_loss', 'dfl_loss')
    return dict(zip(names, torch.as_tensor(t).flatten().tolist()))


def safe(fn):
    """Callback de log/painel NUNCA derruba o treino: erro vira aviso."""
    def wrapped(trainer):
        try:
            fn(trainer)
        except Exception as e:  # noqa: BLE001
            print(f'⚠️ callback {fn.__name__} falhou ({type(e).__name__}: {e}) — treino continua')
    wrapped.__name__ = fn.__name__
    return wrapped


def make_callbacks(target_map: float, cls_weights, events: str = ''):
    """Callbacks que mostram o aprendizado acontecendo + aplicam peso de classe + early-stop."""
    state = {'prev': None}

    def snapshot(trainer):
        return [p.detach().float().cpu().clone() for p in trainer.model.parameters()]

    def on_train_start(trainer):
        state['prev'] = snapshot(trainer)
        filt, state['w1'] = first_layer_filters(trainer)
        emit(events, type='start', epochs=trainer.epochs, device=str(trainer.device), target=target_map, filters=filt)
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
        losses = epoch_losses(trainer)
        loss = ' '.join(f'{k}={v:.3f}' for k, v in losses.items())
        state.update(loss=losses, delta=delta, delta_pct=100 * delta / norm)
        print(f'🧠 época {trainer.epoch + 1}: loss[{loss}] | Δpesos={delta:.4f} ({100 * delta / norm:.2f}% da norma)')

    def on_train_batch_end(trainer):
        if STOP['asked']:
            trainer.stop = True

    def on_fit_epoch_end(trainer):
        if STOP['asked']:
            trainer.stop = True
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
        filt, state['w1'] = first_layer_filters(trainer, state.get('w1'))
        emit(events, type='epoch', epoch=trainer.epoch + 1, epochs=trainer.epochs, verdict=verdict, prev_map50=prev,
             map50=map50, map5095=m.get('metrics/mAP50-95(B)'), precision=m.get('metrics/precision(B)'),
             recall=m.get('metrics/recall(B)'), loss=state.get('loss', {}), delta=state.get('delta'),
             delta_pct=state.get('delta_pct'), filters=filt)
        if map50 >= target_map:
            print(f'✅ mAP50 {map50:.3f} >= {target_map:.3f} — parando')
            emit(events, type='goal', map50=map50)
            trainer.stop = True

    return {'on_train_batch_end': safe(on_train_batch_end), 'on_train_start': safe(on_train_start), 'on_train_epoch_end': safe(on_train_epoch_end),
            'on_fit_epoch_end': safe(on_fit_epoch_end)}


def main():
    args = parse_args()
    root = Path(__file__).parent.parent
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _ask_stop)
    device = select_device(args.device, args.require_cuda)
    print(f'Usando device: {device}')
    apply_vram_limit(device, args.vram_gb)
    train_batch, val_batch = resolve_batch(args.batch, device, args.vram_gb)

    cls_weights = None
    if args.cls_weight:
        cls_weights = (class_weights_from_data(args.data) if args.cls_weight.lower() == 'auto'
                       else [float(w) for w in args.cls_weight.split(',')])
        if cls_weights is None:
            print('⚠️ sem anotações para calcular class weights; ignorando')

    start = args.model or find_start_weight(root, 'yolov8s.pt')
    print(f'🔁 Ponto de partida: {start}')
    model = YOLO(start)
    for event, fn in make_callbacks(args.target_map, cls_weights, args.events).items():
        model.add_callback(event, fn)

    best_pt = root / 'models' / 'best.pt'
    baseline = -1.0
    if best_pt.exists() and not args.skip_validate:
        baseline = validate(best_pt, args.data, args.imgsz, val_batch, device, args.fast_validate)
        print(f'📌 baseline mAP50 = {baseline:.3f}')

    kwargs = dict(data=args.data, epochs=args.max_epochs, imgsz=args.imgsz, batch=train_batch,
                  device=device, name=args.run_name, exist_ok=True, patience=args.patience,
                  cos_lr=True, freeze=args.freeze or None, close_mosaic=10,
                  cache=pick_cache(args.cache, args.ram_gb, args.data, args.imgsz), workers=resolve_workers(args.workers))
    if args.lr0:
        kwargs['lr0'] = args.lr0
    model.train(**kwargs)
    if STOP['asked']:  # parado pelo usuário: pesos já salvos pelo Ultralytics; pula validação/relatório (demoram)
        wdir = Path(model.trainer.save_dir) / 'weights'
        print(f'🛑 treino parado. Pesos salvos em {wdir} (best.pt e last.pt)')
        emit(args.events, type='stopped', path=str(wdir))
        return

    run_best = Path(model.trainer.best)
    if not run_best.exists():
        print('⚠️ best.pt do run não encontrado — veja runs/detect')
        return
    mm = val_metrics(run_best, args.data, args.imgsz, val_batch, device, args.fast_validate)
    new = float(mm.box.map50)
    print(f'📈 mAP50 do run = {new:.3f} (baseline {baseline:.3f})')
    if new > baseline:
        best_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_best, best_pt)
        print(f'📦 models/best.pt atualizado (mAP50={new:.3f})')
    try:  # relatório com matriz de confusão (geral e por classe); nunca derruba o fim do treino
        from report import write_report
        rep = write_report(run_best, args.data, root / 'runs' / 'report', args.imgsz, val_batch, device, map_metrics=mm)
        print(f'📄 Relatório com matriz de confusão: {rep}')
        emit(args.events, type='report', path=str(rep))
    except Exception as e:  # noqa: BLE001
        print(f'⚠️ não foi possível gerar o relatório ({type(e).__name__}: {e}) — rode: make relatorio')


if __name__ == '__main__':
    main()
