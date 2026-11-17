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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001 - sem matplotlib só não gera o gráfico
    plt = None
from ultralytics import YOLO

DEFAULT_RUN_NAME = 'train_iter'
STOP = {'asked': False, 'goal': False, 't': 0.0}
ABORT_DEBOUNCE = 10  # s: 2º sinal dentro deste prazo é ignorado


def stop_reason(asked: bool, goal: bool, done: int, total: int) -> str:
    return 'stopped' if asked else 'goal' if goal else 'epochs' if done >= total else 'patience'


def explain_stop(reason: str, done: int, total: int, patience: int, target: float) -> str:
    """Por que o treino terminou, em português claro (os painéis mostram este texto)."""
    return {
        'epochs': f'Rodou as {total} épocas pedidas até o fim.',
        'goal': f'Parou na época {done} de {total}: o mAP50 chegou ao alvo de {target:.2f}. Para treinar mais, suba o alvo (--target-map).',
        'patience': f'Parou na época {done} de {total}: {patience} épocas seguidas sem melhorar (patience). O modelo parou de evoluir; '
                    f'aumente --patience para insistir mais.',
        'stopped': f'Você parou o treino na época {done} de {total}. Os pesos ficaram salvos em best.pt e last.pt.',
    }[reason]


def _ask_stop(signum, frame):
    """q / Ctrl+C: termina a época atual e salva best.pt/last.pt (2ª vez = aborta na hora)."""
    now = time.monotonic()
    if STOP['asked']:
        if now - STOP['t'] > ABORT_DEBOUNCE:  # sinal repetido logo após o 1º (duplo clique, Ctrl+C + SIGTERM) não aborta
            raise KeyboardInterrupt
        return
    STOP['asked'], STOP['t'] = True, now
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


def ckpt_info(path):
    """Última estatística gravada no .pt de treino do Ultralytics (train_metrics: mAP50 do melhor epoch), ou None."""
    try:
        ck = torch.load(str(path), map_location='cpu', weights_only=False)  # arquivo local do próprio projeto
        m50 = (ck.get('train_metrics') or {}).get('metrics/mAP50(B)')
        return {'map50': float(m50)} if m50 is not None else None
    except Exception:  # noqa: BLE001
        return None


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


def save_epoch_plot(run_dir, history):
    """epoch_metrics.json + epoch_metrics.png (precisão, recall e mAP50 por época), atualizados a cada época."""
    run_dir = Path(run_dir)
    (run_dir / 'epoch_metrics.json').write_text(json.dumps(history, indent=1), encoding='utf-8')
    if plt is None or not history:
        return None
    xs = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label, mk in (('precision', 'precisão', 'o'), ('recall', 'recall', 's'), ('map50', 'mAP50', 'x')):
        ax.plot(xs, [h.get(key) or 0 for h in history], marker=mk, label=label)
    ax.set_xlabel('época')
    ax.set_ylabel('valor (0 a 1)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=.3)
    ax.legend()
    ax.set_title('Precisão, recall e mAP50 por época')
    fig.tight_layout()
    out = run_dir / 'epoch_metrics.png'
    fig.savefig(out)
    plt.close(fig)
    return out


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


def make_callbacks(target_map: float, cls_weights, events: str = '', baseline=None, resumed=None, patience=None):
    """Callbacks que mostram o aprendizado acontecendo + aplicam peso de classe + early-stop."""
    goal_on = not (baseline and baseline >= target_map)   # alvo já superado na partida: ignorá-lo (senão pararia na 1ª época)
    state = {'prev': None, 'map50': baseline if baseline and baseline > 0 else None}   # 1ª época já é comparada ao ponto de partida

    def snapshot(trainer):
        return [p.detach().float().cpu().clone() for p in trainer.model.parameters()]

    def on_train_start(trainer):
        state['prev'] = snapshot(trainer)
        print(f'📁 Run em {trainer.save_dir} — best.pt e last.pt são gravados em weights/ a cada época')
        filt, state['w1'] = first_layer_filters(trainer)
        emit(events, type='start', epochs=trainer.epochs, device=str(trainer.device), target=target_map, filters=filt,
             baseline=baseline, resumed_from=resumed, patience=patience, goal_on=goal_on,
             run_dir=str(trainer.save_dir))
        if not goal_on:
            print(f'ℹ️ alvo {target_map:.2f} já superado pelo ponto de partida (mAP50 {baseline:.3f}): o treino roda até acabarem as épocas '
                  f'ou até {patience} épocas sem melhorar. Suba --target-map para parar antes.')
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
        state.setdefault('hist', []).append(dict(epoch=trainer.epoch + 1, map50=map50, precision=m.get('metrics/precision(B)'),
                                                 recall=m.get('metrics/recall(B)'), map5095=m.get('metrics/mAP50-95(B)')))
        save_epoch_plot(trainer.save_dir, state['hist'])
        if goal_on and map50 >= target_map:
            STOP['goal'] = True
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
    info = ckpt_info(start)
    continuing = info is not None and Path(start).name in ('best.pt', 'last.pt')   # modelo já treinado por nós
    if continuing:
        print(f'📊 Última estatística gravada nele: mAP50 = {info["map50"]:.3f} — o treino parte daí para melhorar')
    model = YOLO(start)

    best_pt = root / 'models' / 'best.pt'
    baseline = -1.0
    if best_pt.exists() and not args.skip_validate:   # revalida nos dados atuais: é o número a superar
        baseline = validate(best_pt, args.data, args.imgsz, val_batch, device, args.fast_validate)
        print(f'📌 baseline mAP50 = {baseline:.3f}')
    seed = baseline if baseline >= 0 else (info or {}).get('map50')
    for event, fn in make_callbacks(args.target_map, cls_weights, args.events, baseline=seed,
                                    resumed=start if continuing else None, patience=args.patience).items():
        model.add_callback(event, fn)

    kwargs = dict(data=args.data, epochs=args.max_epochs, imgsz=args.imgsz, batch=train_batch,
                  project=str(root / 'runs' / 'detect'),   # sem isso o Ultralytics grava em <raiz do repo>/runs, fora do ia-service
                  device=device, name=args.run_name, exist_ok=True, patience=args.patience,
                  cos_lr=True, freeze=args.freeze or None, close_mosaic=10,
                  cache=pick_cache(args.cache, args.ram_gb, args.data, args.imgsz), workers=resolve_workers(args.workers))
    if continuing:  # refinar modelo já treinado: LR baixo e warmup curto, p/ não "esquecer" o que já aprendeu
        kwargs.update(optimizer='AdamW', lr0=args.lr0 or 0.0005, warmup_epochs=1.0)
        print(f'🎯 Refinando modelo treinado: AdamW lr0={kwargs["lr0"]} (mude com --lr0)')
    elif args.lr0:
        kwargs['lr0'] = args.lr0
    print(f'📅 {args.max_epochs} épocas pedidas · para antes se mAP50 ≥ {args.target_map:.2f} ou {args.patience} épocas sem melhorar')
    aborted = False
    try:
        model.train(**kwargs)
    except KeyboardInterrupt:   # 2º sinal de parada (ex: durante a validação final): os pesos já estão salvos em weights/
        aborted = True
        print('🛑 abortado na validação final — os pesos do treino já estavam salvos')
    tr = model.trainer
    done, total = tr.epoch + 1, tr.epochs
    if aborted:
        STOP['asked'] = True   # cai no caminho "parado pelo usuário" abaixo
    reason = stop_reason(STOP['asked'], STOP['goal'], done, total)
    why = explain_stop(reason, done, total, args.patience, args.target_map)
    print(f'🏁 {why}')
    emit(args.events, type='end', reason=reason, epochs_done=done, epochs_total=total, text=why)
    wdir = Path(tr.save_dir) / 'weights'   # confere (e mostra) o que realmente foi salvo
    print('💾 ' + ' · '.join(f'{n} ' + (f'OK ({(wdir / n).stat().st_size / 1e6:.1f} MB)' if (wdir / n).exists() else 'NÃO GERADO')
                             for n in ('best.pt', 'last.pt')) + f'  em {wdir}')
    if STOP['asked']:  # parado pelo usuário: pesos já salvos pelo Ultralytics; pula validação/relatório (demoram)
        models_dir = root / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        if (wdir / 'last.pt').exists():
            shutil.copy2(wdir / 'last.pt', models_dir / 'last.pt')
        if (wdir / 'best.pt').exists() and not best_pt.exists():   # nunca sobrescreve um best.pt já validado (sem mAP aqui p/ comparar)
            shutil.copy2(wdir / 'best.pt', best_pt)
            print('📦 models/best.pt criado a partir deste run (ainda não validado; make relatorio confere)')
        print(f'🛑 treino parado. Pesos salvos em {wdir} (best.pt e last.pt) · cópia em models/last.pt')
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
        rep = write_report(run_best, args.data, root / 'runs' / 'report', args.imgsz, val_batch, device, map_metrics=mm,
                           run_dir=Path(tr.save_dir))
        print(f'📄 Relatório com matriz de confusão: {rep}')
        emit(args.events, type='report', path=str(rep))
    except Exception as e:  # noqa: BLE001
        print(f'⚠️ não foi possível gerar o relatório ({type(e).__name__}: {e}) — rode: make relatorio')


if __name__ == '__main__':
    main()
