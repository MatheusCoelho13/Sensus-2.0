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

lock = threading.Lock()
proc: 'subprocess.Popen | None' = None
last_cmd = ''


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
    if not re.fullmatch(r'[\w./-]+\.pt', model) or '..' in model or model.startswith('/'):
        raise ValueError('model deve ser um caminho relativo .pt (ex: models/yolov8n.pt)')
    return dict(target=num('target', float, 0.05, 0.99, 0.6), epochs=num('max_epochs', int, 1, 1000, 300),
                imgsz=num('imgsz', int, 320, 1280, 640), batch=num('batch', int, 1, 128, 16), model=model)


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
                 '--patience', '100', '--target-map', str(a['target']), '--events', str(EVENTS)]
        cmd = shlex.join(train)
        if not DATASET.exists():  # baixa/balanceia o COCO antes se ainda não existe
            cmd = shlex.join([py, 'scripts/build_coco_dataset.py']) + ' && ' + cmd
        last_cmd = cmd
        log = LOG.open('wb')
        proc = subprocess.Popen(['sh', '-c', cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})


def stop():
    with lock:
        if running():
            os.killpg(proc.pid, signal.SIGTERM)


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
    return dict(running=running(), exit_code=None if proc is None or running() else proc.returncode,
                dataset_ready=DATASET.exists(), events=read_events(), log=read_log())


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
