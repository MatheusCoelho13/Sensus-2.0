#!/usr/bin/env python3
"""Sensores do sistema para a TUI (stdlib pura): CPU, RAM, GPU NVIDIA e temperaturas.

Amostra numa thread própria (nunca bloqueia o desenho da TUI). Linux/WSL.
Sem sensor disponível o valor fica None (ex: o WSL não expõe a temperatura da CPU).
"""
import glob
import os
import shutil
import subprocess
import threading
import time
from collections import deque

HIST = 120  # amostras guardadas por série (~2 min a 1 Hz)
SMI_FIELDS = 'name,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,power.limit,clocks.sm'
CPU_HWMON = ('coretemp', 'k10temp', 'zenpower', 'cpu_thermal', 'acpitz')


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ''


def _num(s: str):
    try:
        return float(s)
    except ValueError:
        return None   # '[N/A]', '[Not Supported]'...


def cpu_temp():
    """Maior temperatura de CPU (°C) entre os sensores conhecidos; None se o sistema não expõe."""
    temps = []
    for d in glob.glob('/sys/class/hwmon/hwmon*'):
        if _read(d + '/name') in CPU_HWMON:
            temps += [int(v) / 1000 for v in (_read(f) for f in glob.glob(d + '/temp*_input')) if v.lstrip('-').isdigit()]
    if not temps:
        for z in glob.glob('/sys/class/thermal/thermal_zone*'):
            t = _read(z + '/type').lower()
            v = _read(z + '/temp')
            if ('x86_pkg' in t or 'cpu' in t or 'soc' in t) and v.lstrip('-').isdigit():
                temps.append(int(v) / 1000)
    temps = [t for t in temps if 0 < t < 150]
    return max(temps) if temps else None


def _cpu_times():
    """{'cpu': (idle, total), 'cpu0': ...} a partir de /proc/stat."""
    out = {}
    for ln in _read('/proc/stat').splitlines():
        p = ln.split()
        if p and p[0].startswith('cpu'):
            v = [int(x) for x in p[1:] if x.isdigit()]
            if len(v) >= 5:
                out[p[0]] = (v[3] + v[4], sum(v[:8]))   # idle + iowait / total
    return out


def _ram():
    mem = {}
    for ln in _read('/proc/meminfo').splitlines():
        k, _, v = ln.partition(':')
        if v.split():
            mem[k] = int(v.split()[0]) / 1048576   # kB -> GB
    total = mem.get('MemTotal')
    return (total - mem['MemAvailable'], total) if total and 'MemAvailable' in mem else (None, None)


class SysMon:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}
        self.hist = {k: deque(maxlen=HIST) for k in ('cpu', 'ram', 'ctemp', 'gpu', 'gtemp', 'vram')}
        self._prev, self._gpu, self._gpu_at = None, None, 0.0
        self._smi = shutil.which('nvidia-smi') or next((p for p in ('/usr/lib/wsl/lib/nvidia-smi',) if os.path.exists(p)), None)
        threading.Thread(target=self._loop, daemon=True).start()

    def _sample_gpu(self):
        if not self._smi or time.time() - self._gpu_at < 2.0:   # nvidia-smi custa ~100ms: a cada 2s
            return
        self._gpu_at = time.time()
        try:
            r = subprocess.run([self._smi, f'--query-gpu={SMI_FIELDS}', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=4)
            f = [x.strip() for x in r.stdout.splitlines()[0].split(',')]
            self._gpu = dict(name=f[0], util=_num(f[1]), temp=_num(f[2]), mem_used=_num(f[3]), mem_total=_num(f[4]),
                             power=_num(f[5]), power_limit=_num(f[6]), clock=_num(f[7])) if r.returncode == 0 and len(f) >= 8 else None
        except (OSError, subprocess.SubprocessError, IndexError):
            self._gpu = None

    def _loop(self):
        while True:
            now, cores, cpu = _cpu_times(), [], None
            if self._prev and 'cpu' in now:
                def pct(k):
                    di, dt = now[k][0] - self._prev[k][0], now[k][1] - self._prev[k][1]
                    return 100 * (1 - di / dt) if dt > 0 else 0.0
                cpu = pct('cpu')
                cores = [pct(k) for k in sorted((k for k in now if k != 'cpu' and k in self._prev), key=lambda k: int(k[3:]))]
            self._prev = now
            used, total = _ram()
            ct = cpu_temp()
            self._sample_gpu()
            g = self._gpu
            with self.lock:
                self.data = dict(cpu=cpu, cores=cores, ram_used=used, ram_total=total, cpu_temp=ct, gpu=g)
                for key, val in (('cpu', cpu), ('ram', 100 * used / total if used and total else None), ('ctemp', ct),
                                 ('gpu', g and g['util']), ('gtemp', g and g['temp']),
                                 ('vram', 100 * g['mem_used'] / g['mem_total'] if g and g['mem_used'] is not None and g['mem_total'] else None)):
                    if val is not None:
                        self.hist[key].append(val)
            time.sleep(1.0)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(data=dict(self.data), hist={k: list(v) for k, v in self.hist.items()})
