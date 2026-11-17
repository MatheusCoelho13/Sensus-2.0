#!/usr/bin/env python3
"""Painel gráfico do treino (stdlib pura, sem dependências).

  make painel   ->  http://127.0.0.1:8765

Sobe/para o scripts/train.py como subprocesso. O train.py escreve um evento por
época em runs/ui/events.jsonl (--events) e o painel faz polling em /api/state.
Escuta só em 127.0.0.1.
"""
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / 'runs' / 'ui'
EVENTS, LOG = RUN_DIR / 'events.jsonl', RUN_DIR / 'train.log'
PAGE = ROOT / 'static' / 'treino.html'
DATASET = ROOT / 'datasets' / 'coco_nav' / 'images' / 'train'
REPORT = ROOT / 'runs' / 'report' / 'latest.html'

lock = threading.Lock()
proc: 'subprocess.Popen | None' = None
last_cmd = ''


def run_dir(events=None) -> Path:
    """Pasta do run atual: a do último evento 'start' (o train.py informa), senão runs/detect/train_iter."""
    for e in reversed(events or []):
        if e.get('type') == 'start' and e.get('run_dir'):
            return Path(e['run_dir'])
    return ROOT / 'runs' / 'detect' / 'train_iter'


def list_images(events=None) -> list:
    from report import IMAGES   # mesma lista/legendas do relatório
    d, seen, out = run_dir(events), set(), []
    for n, c in IMAGES:
        if (d / n).exists() and c not in seen:   # variantes (Box...) têm a mesma legenda: mostra só a que existe
            seen.add(c)
            out.append(dict(name=n, caption=c, v=int((d / n).stat().st_mtime)))
    return out


def list_files(events=None) -> list:
    """Arquivos que o run deixa na pasta: pesos (best/last), métricas e argumentos."""
    d = run_dir(events)
    out = []
    for rel in ('weights/best.pt', 'weights/last.pt', 'results.csv', 'args.yaml'):
        p = d / rel
        if p.exists():
            try:
                shown = str(p.relative_to(ROOT))
            except ValueError:
                shown = str(p)
            out.append(dict(name=rel, size=p.stat().st_size, path=shown))
    return out


def image_path(name: str, events=None):
    """Só serve arquivos da lista IMAGES (sem path traversal)."""
    from report import IMAGES
    p = run_dir(events) / name
    return p if name in {n for n, _ in IMAGES} and p.exists() else None


def write_gallery() -> 'Path | None':
    """Copia as imagens do run para runs/ui/imagens/ (abra a pasta no VS Code) — tecla i da TUI. Devolve a pasta."""
    import shutil
    ev = read_events()
    imgs = list_images(ev)
    if not imgs:
        return None
    d = run_dir(ev)
    out = RUN_DIR / 'imagens'
    out.mkdir(parents=True, exist_ok=True)
    for i in imgs:
        src = d / i['name']
        if src.exists():
            shutil.copy2(src, out / Path(i['name']).name)
    return out


def list_models() -> list:
    """Todo .pt dentro de models/ (best.pt primeiro). Atualiza a cada poll: basta soltar o arquivo lá."""
    found = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'models').glob('*.pt'))
    return sorted(found, key=lambda m: (not m.endswith('/best.pt'), m))


def running() -> bool:
    return proc is not None and proc.poll() is None


def validate(body: dict) -> dict:
    """Valida o payload do /api/start; levanta ValueError com mensagem p/ o usuário."""
    def num(key, typ, lo, hi, default):
        try:
            v = typ(body.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f'{key} inválido')
        if not lo <= v <= hi:
            raise ValueError(f'{key} deve estar entre {lo} e {hi}')
        return v
    model = str(body.get('model', 'models/yolov8n.pt'))
    if not re.fullmatch(r'[\w ./-]+\.pt', model) or '..' in model or model.startswith('/'):
        raise ValueError('model deve ser um caminho relativo .pt (ex: models/yolov8n.pt)')
    return dict(target=num('target', float, 0.05, 0.99, 0.6), epochs=num('max_epochs', int, 1, 1000, 300),
                imgsz=num('imgsz', int, 320, 1280, 640), batch=num('batch', int, 0, 128, 0), ram_gb=num('ram_gb', float, 0, 1024, 0), patience=num('patience', int, 1, 1000, DEFAULT_PATIENCE),
                vram_gb=num('vram_gb', float, 0, 256, 0), model=model)


def start(body: dict):
    global proc, last_cmd
    a = validate(body)
    with lock:
        if running():
            raise ValueError('já existe um treino rodando')
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        EVENTS.write_text('')
        py = sys.executable
        train = [py, 'scripts/train.py', '--data', 'config/data.yaml', '--model', a['model'], '--imgsz', str(a['imgsz']),
                 '--batch', str(a['batch']), '--cls-weight', 'auto', '--max-epochs', str(a['epochs']),
                 '--patience', str(a['patience']), '--target-map', str(a['target']), '--events', str(EVENTS),
                 '--ram-gb', str(a['ram_gb']), '--vram-gb', str(a['vram_gb'])]
        cmd = 'exec ' + shlex.join(train)   # exec: o pid do subprocesso passa a ser o do python
        if not DATASET.exists():  # baixa/balanceia o COCO antes se ainda não existe
            cmd = shlex.join([py, 'scripts/build_coco_dataset.py']) + ' && ' + cmd
        last_cmd = cmd
        log = LOG.open('wb')
        proc = subprocess.Popen(['sh', '-c', cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})


DEFAULT_PATIENCE = 100  # épocas sem melhorar antes do treino parar sozinho
STOP_GRACE = 180  # s p/ o treino salvar os pesos antes de ser morto à força


def kill():
    if running():
        os.killpg(proc.pid, signal.SIGKILL)


def stop():
    """SIGTERM: o train.py termina a época e salva best.pt/last.pt. Se não sair em STOP_GRACE s, SIGKILL."""
    with lock:
        if running():
            os.kill(proc.pid, signal.SIGTERM)  # só o processo principal: killpg matava os workers do DataLoader junto
            t = threading.Timer(STOP_GRACE, kill)
            t.daemon = True
            t.start()


def read_events() -> list:
    out = []
    if EVENTS.exists():
        for ln in EVENTS.read_text(encoding='utf-8', errors='ignore').splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass  # linha cortada (escrita em andamento)
    for ev in out[:-1]:  # tiles dos filtros só do último evento (payload pequeno)
        ev.pop('filters', None)
    return out


def read_log(n: int = 60) -> list:
    if not LOG.exists():
        return []
    with LOG.open('rb') as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 16384))
        txt = f.read().decode('utf-8', 'ignore')
    return [l for l in re.split(r'[\r\n]+', re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', txt)) if l.strip()][-n:]


def state() -> dict:
    ev = read_events()
    return dict(running=running(), exit_code=None if proc is None or running() else proc.returncode,
                dataset_ready=DATASET.exists(), report_ready=REPORT.exists(), models=list_models(), events=ev, images=list_images(ev), files=list_files(ev), run_dir=str(run_dir(ev)), log=read_log())


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/api/state':
            return self._json(state())
        if self.path in ('/', '/index.html'):
            data = PAGE.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        if self.path.startswith('/img/'):
            p = image_path(self.path[5:].split('?')[0], read_events())
            if p:
                data = p.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'image/png' if p.suffix == '.png' else 'image/jpeg')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                return self.wfile.write(data)
        if self.path == '/relatorio' and REPORT.exists():
            data = REPORT.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        self._json({'error': 'not found'}, 404)

    def do_POST(self):
        try:
            n = int(self.headers.get('Content-Length') or 0)
            body = json.loads(self.rfile.read(min(n, 4096)) or b'{}')
            if self.path == '/api/start':
                start(body)
            elif self.path == '/api/stop':
                stop()
            else:
                return self._json({'error': 'not found'}, 404)
            self._json({'ok': True})
        except ValueError as e:
            self._json({'error': str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self._json({'error': f'falha interna: {e}'}, 500)

    def log_message(self, *a):  # silencia o log de cada request
        pass


def main():
    port = int(os.environ.get('PORT', 8765))
    srv = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    url = f'http://127.0.0.1:{port}'
    print(f'🧪 Painel de treino em {url}  (Ctrl+C sai; o treino em andamento é encerrado)')
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop()


if __name__ == '__main__':
    main()
