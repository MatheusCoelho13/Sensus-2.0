#!/usr/bin/env python3
"""Painel do treino NO TERMINAL (curses, stdlib pura).

  make tui

Mesmo motor do painel web (scripts/train_ui.py): sobe o train.py, lê runs/ui/events.jsonl.
Teclas: s iniciar/parar · +/- alvo mAP50 · e épocas · z imgsz · p patience · a LR · f freeze · b batch · r RAM · v VRAM · u AMP · c cache · w workers · m modelo · o relatório · l log
        t sensores (CPU/RAM/GPU/temperaturas, estilo btop) · T abre o btop de verdade · q sair (para o treino)
"""
import curses
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sysmon  # noqa: E402
import train_ui as core  # noqa: E402

GLYPH = {
    '0': ['█████', '█   █', '█   █', '█   █', '█████'], '1': ['  █  ', ' ██  ', '  █  ', '  █  ', '█████'],
    '2': ['█████', '    █', '█████', '█    ', '█████'], '3': ['█████', '    █', '█████', '    █', '█████'],
    '4': ['█   █', '█   █', '█████', '    █', '    █'], '5': ['█████', '█    ', '█████', '    █', '█████'],
    '6': ['█████', '█    ', '█████', '█   █', '█████'], '7': ['█████', '    █', '   █ ', '  █  ', '  █  '],
    '8': ['█████', '█   █', '█████', '█   █', '█████'], '9': ['█████', '█   █', '█████', '    █', '█████'],
    '.': ['  ', '  ', '  ', '██', '██'],
}
EIGHTHS = ' ▁▂▃▄▅▆▇█'
CONFIG_KEYS = {ord(c) for c in 'eEbBrRvVmMzZpPaAfFuUcCwW+=-_'}
HELP = [
    ('ÉPOCAS (tecla e)', 'Quantas vezes a rede vê o conjunto de treino inteiro. Mais épocas = mais aprendizado e mais tempo. '
                         'É um LIMITE: o treino pode parar antes (veja "quando para").'),
    ('QUANDO PARA', '1) acabaram as épocas · 2) o mAP50 chegou ao alvo · 3) passaram N épocas seguidas sem melhorar (patience) · '
                    '4) você apertou s ou q. Ao fim, o painel diz qual foi o motivo.'),
    ('ALVO mAP50 (+/-)', 'Nota de 0 a 1 de quão bem a rede acha os objetos. Chegou nela, o treino encerra. Se o modelo de partida '
                         '(best.pt) já passa do alvo, o alvo é ignorado e o treino segue até acabar as épocas.'),
    ('IMGSZ (z)', 'Resolução das imagens no treino (px). Maior enxerga melhor objetos pequenos (book, bottle, cell phone), mas gasta '
                  'muito mais VRAM e tempo: 640 padrão, 800, 960, 1280. Na GPU de 4GB, com 960+ use batch pequeno (4).'),
    ('PATIENCE (p)', 'Quantas épocas seguidas sem melhorar antes de parar sozinho. Menor = para antes (economiza tempo).'),
    ('LR (a)', 'Taxa de aprendizado inicial. "auto" = 0.002 ao partir de pesos pré-treinados (0.0005 se continua de best.pt). '
               'Alta demais faz o modelo esquecer o que já sabia (o mAP chegou a cair de 0.50 p/ 0.13).'),
    ('FREEZE (f)', 'Camadas iniciais congeladas (não treinam). "auto" = 10 ao partir de pesos pré-treinados. 0 treina tudo.'),
    ('BATCH (b)', 'Imagens processadas por vez. Maior = mais rápido, mas usa mais memória da GPU. "máx" ocupa 90% da VRAM.'),
    ('RAM (r) / VRAM (v)', 'RAM: quanto de memória pode guardar as imagens em cache. VRAM: quanto da placa de vídeo o treino pode usar. '
                           '"máx" = sem limite.'),
    ('AMP (u)', 'Precisão mista (fp16): mais rápido e usa menos VRAM. Em algumas GTX 16xx dá loss nan — se acontecer, desligue (off).'),
    ('CACHE (c)', 'ram = imagens decodificadas na memória (mais rápido; só liga se couber no limite de RAM) · disk = em disco · off = sem cache.'),
    ('WORKERS (w)', 'Processos que carregam e aumentam as imagens enquanto a GPU treina. "máx" = todas as threads da CPU. Se a GPU ficar ociosa, aumente.'),
    ('MODELO (m)', 'Ponto de partida. best.pt = continua do melhor treino já feito e tenta melhorar (taxa de aprendizado baixa).'),
    ('▲ RECOMPENSA / ▼ PUNIÇÃO', 'O mAP50 subiu / caiu em relação à época anterior. A rede corrige os pesos a partir do erro (loss) '
                                 'contra os rótulos: é aprendizado supervisionado.'),
    ('IMPORTANTE', 'Mudar a configuração não afeta um treino em andamento: pare (s) e inicie de novo.'),
]
IMGSZ_CHOICES = [640, 800, 960, 1280]
LR_CHOICES = [0, 0.0005, 0.001, 0.002, 0.005, 0.01]   # 0 = auto
FREEZE_CHOICES = [-1, 0, 10, 15, 20]                   # -1 = auto
BATCH_CHOICES = [0, 4, 8, 16, 32, 64]   # 0 = máximo (ocupa a GPU)
RAM_CHOICES = [0, 2, 4, 8, 16, 32]       # GB p/ cache de imagens; 0 = máximo (toda a RAM livre)
VRAM_CHOICES = [0, 2, 3, 4, 6, 8, 12]    # GB de GPU; 0 = máximo (a GPU inteira)
AMP_CHOICES = ['on', 'off']
CACHE_CHOICES = ['ram', 'disk', 'false']
WORKERS_CHOICES = [0, 2, 4, 6, 8, 12]    # 0 = máximo (todas as threads da CPU)

INK, REWARD, PUNISH, DIM, L1, L2, L3 = range(1, 8)

# Parada automática por temperatura da GPU (°C). 0 desliga. Ex: MAX_TEMP=80 make tui
try:
    MAX_TEMP = int(os.environ.get('MAX_TEMP', '85'))
except ValueError:
    MAX_TEMP = 85
try:
    MAX_CPU_TEMP = int(os.environ.get('MAX_CPU_TEMP', '95'))   # só vale se a CPU tiver sensor (Linux nativo ou LibreHardwareMonitor)
except ValueError:
    MAX_CPU_TEMP = 95
HOT_SAMPLES = 2  # leituras seguidas >= MAX_TEMP (a GPU é lida a cada 2s) — ignora picos de 1 amostra


def rgb_to_xterm(r, g, b):
    if max(r, g, b) - min(r, g, b) < 12:  # cinza -> rampa 232-255
        return 232 + min(23, int(r / 256 * 24))
    return 16 + 36 * round(r / 255 * 5) + 6 * round(g / 255 * 5) + round(b / 255 * 5)


def upscale(t, k, n):
    """k×k RGB -> n×n RGB por interpolação bilinear: neurônio liso em vez de pixels enormes."""
    out = []
    for y in range(n):
        sy = min(max((y + .5) * k / n - .5, 0), k - 1)
        y0 = int(sy); y1 = min(y0 + 1, k - 1); fy = sy - y0
        row = []
        for x in range(n):
            sx = min(max((x + .5) * k / n - .5, 0), k - 1)
            x0 = int(sx); x1 = min(x0 + 1, k - 1); fx = sx - x0
            row.append(tuple(round((t[y0][x0][c] * (1 - fx) + t[y0][x1][c] * fx) * (1 - fy)
                                   + (t[y1][x0][c] * (1 - fx) + t[y1][x1][c] * fx) * fy) for c in range(3)))
        out.append(row)
    return out


def digits_rows(value: float) -> list:
    """'0.485' em 5 linhas de blocos."""
    txt = f'{value:.3f}'
    return [' '.join(GLYPH[ch][r] for ch in txt) for r in range(5)]


def bar_cells(v: float, rows: int) -> list:
    """Coluna de barra vertical: lista (de baixo p/ cima) de caracteres, resolução de 1/8 de linha."""
    h = max(0.0, min(1.0, v)) * rows
    full, frac = int(h), h - int(h)
    cells = ['█'] * full
    if full < rows:
        cells.append(EIGHTHS[int(frac * 8)] if frac >= 1 / 8 else ' ')
    return cells + [' '] * (rows - len(cells))


def spark(values: list, width: int) -> str:
    vs = values[-width:]
    mx = max(vs + [1e-9])
    return ''.join(EIGHTHS[max(1, min(8, round(8 * v / mx)))] for v in vs)


def spark_abs(values: list, width: int, top: float = 100.0) -> str:
    """Sparkline numa escala fixa 0..top (percentuais/temperaturas não podem ser relativos ao máximo)."""
    return ''.join(EIGHTHS[max(1, min(8, round(8 * v / top)))] for v in values[-width:])


def temp_pair(t):
    """Cor da temperatura: verde < 70°C, âmbar < 80°C, vermelho acima."""
    return DIM if t is None else REWARD if t < 70 else L1 if t < 80 else PUNISH


class UI:
    def __init__(self, scr):
        self.scr, self.msg, self.show_log = scr, '', False
        self.msg_until, self.show_help, self.show_sys = 0.0, False, False
        self.mon = sysmon.SysMon()
        self.hot, self.hot_seen, self.thermal_stop = 0, None, False
        self.ask, self.busy, self.was_running, self.msg_pair = None, None, False, PUNISH   # ask: 'report' | 'publish' | None
        self.target, self.epochs, self.model = 0.60, 300, ''
        self.batch_i, self.ram_i, self.vram_i = 0, 0, 0
        self.amp_i, self.cache_i, self.workers_i = 0, 0, 0
        self.imgsz_i, self.lr_i, self.freeze_i, self.patience = 0, 0, 0, core.DEFAULT_PATIENCE
        self.quitting = False
        self.S, self.last_poll, self.pairs = {'running': False, 'events': [], 'log': [], 'dataset_ready': True, 'models': [], 'report_ready': False, 'files': [],
                                              'exit_code': None}, 0.0, {}
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        self.c256 = curses.COLORS >= 256
        cols = {INK: 255, REWARD: 154, PUNISH: 203, DIM: 245, L1: 179, L2: 110, L3: 166} if self.c256 else {
            INK: 7, REWARD: 2, PUNISH: 1, DIM: 7, L1: 3, L2: 6, L3: 5}
        for k, c in cols.items():
            curses.init_pair(k, c, -1)
        self.next_pair = 8
        scr.timeout(250)

    # ---------- util ----------
    def put(self, y, x, s, pair=INK, bold=False):
        h, w = self.scr.getmaxyx()
        if 0 <= y < h and 0 <= x < w:
            try:
                self.scr.addnstr(y, x, s, max(0, w - x), curses.color_pair(pair) | (curses.A_BOLD if bold else 0))
            except curses.error:
                pass

    def bg_pair(self, rgb):
        idx = rgb_to_xterm(*rgb)
        if idx not in self.pairs:
            if self.next_pair >= curses.COLOR_PAIRS:
                return INK
            curses.init_pair(self.next_pair, 0, idx)
            self.pairs[idx] = self.next_pair
            self.next_pair += 1
        return self.pairs[idx]

    def half(self, top, bot):
        """(char, pair) p/ um par de pixels (cima, baixo); None = fora do orbe (fundo do terminal)."""
        if top is None and bot is None:
            return None
        if top is None:
            return '▄', self.fgbg_pair(bot, None)
        return '▀', self.fgbg_pair(top, bot)

    def fgbg_pair(self, fg, bg):
        key = (rgb_to_xterm(*fg), rgb_to_xterm(*bg) if bg else -1)
        if key not in self.pairs:
            if self.next_pair >= curses.COLOR_PAIRS:
                return INK
            curses.init_pair(self.next_pair, key[0], key[1])
            self.pairs[key] = self.next_pair
            self.next_pair += 1
        return self.pairs[key]

    def draw_neurons(self, f, y, h, w):
        """Cada neurônio = filtro 3×3 suavizado, desenhado com meio-bloco (▀: cor de cima + cor de baixo).
        Abaixo de cada um, uma barrinha mostra o quanto ele mudou nesta época."""
        k, tiles, delta = f['k'], f['tiles'], f.get('delta')
        n, avail = len(tiles), h - 4 - y
        rich = self.c256 and curses.COLOR_PAIRS >= 4096
        size = 8
        if rich:
            for cand in (8, 6, 4):
                per = max(1, (min(w, 72) - 2) // (cand + 1))
                size = cand
                if -(-n // per) * (cand // 2 + 1) <= avail:
                    break
        tw = size if rich else k * 2          # largura do neurônio em colunas
        th = size // 2 if rich else k         # altura em linhas
        per = max(1, (min(w, 72) - 2) // (tw + 1))
        dmax = max(delta or [0]) or 1e-12
        for i, t in enumerate(tiles):
            ty, tx = y + (i // per) * (th + 1), 2 + (i % per) * (tw + 1)
            if ty + th + 1 > h - 3:
                break
            if rich:
                px, c = upscale(t, k, size), (size - 1) / 2
                ins = lambda x, yy: (x - c) ** 2 + (yy - c) ** 2 <= (size / 2) ** 2   # recorta os cantos: vira orbe
                for r in range(th):
                    for x in range(tw):
                        cell = self.half(px[2 * r][x] if ins(x, 2 * r) else None, px[2 * r + 1][x] if ins(x, 2 * r + 1) else None)
                        if cell:
                            self.put(ty + r, tx + x, *cell)
            else:
                for yy in range(k):
                    for xx in range(k):
                        if self.c256:
                            self.put(ty + yy, tx + xx * 2, '  ', self.bg_pair(t[yy][xx]))
                        else:
                            self.put(ty + yy, tx + xx * 2, '░░▒▒▓▓██'[int(sum(t[yy][xx]) / 765 * 7)] * 2, DIM)
            if delta:
                self.put(ty + th, tx, '▁' * tw, DIM)
                self.put(ty + th, tx, '▁' * max(1, round(tw * delta[i] / dmax)), REWARD)

    def rule(self, y, label=''):
        h, w = self.scr.getmaxyx()
        self.put(y, 0, '─' * (w - 1), DIM)
        if label:
            self.put(y, 2, f' {label} ', DIM)

    # ---------- ações ----------
    def say(self, text, secs=6, pair=PUNISH):
        self.msg, self.msg_until, self.msg_pair = text, time.time() + secs, pair

    # ---------- pós-treino: relatório + publicar no GitHub (sempre perguntando) ----------
    PROMPTS = {'report': 'Treino acabou. Gerar o relatório com matriz de confusão por classe?  [s] sim  [n] não',
               'publish': 'Commitar e subir esta run no GitHub (só models/best.pt + o relatório, nada além)?  [s] sim  [n] não'}

    def after_training(self):
        """Chamado quando o treino acaba: pergunta o que fazer. O train.py já gera o relatório se terminou por conta própria."""
        has_weights = any(f['name'] == 'weights/best.pt' for f in self.S.get('files') or [])
        if self.quitting or not has_weights:
            return
        made_report = any(e['type'] == 'report' for e in self.S['events'])
        self.ask = 'publish' if made_report else 'report'

    def answer(self, yes):
        stage, self.ask = self.ask, None
        if stage == 'report':
            if not yes:
                self.ask = 'publish'
                return
            w = core.ROOT / 'models' / 'best.pt'
            if not w.exists():
                w = core.run_dir(self.S['events']) / 'weights' / 'best.pt'
            self.run_post('relatório', [sys.executable, 'scripts/report.py', '--weights', str(w),
                                        '--data', 'config/data.yaml', '--imgsz', str(IMGSZ_CHOICES[self.imgsz_i])], next_ask='publish')
        elif stage == 'publish':
            if yes:
                self.run_post('publicação no GitHub', [sys.executable, 'scripts/publish.py'])
            else:
                self.say('ok: nada foi enviado. Quando quiser: make publicar', 10, DIM)

    def run_post(self, label, cmd, next_ask=None):
        """Roda o comando numa thread (a TUI não trava) e avisa o resultado."""
        self.busy = f'{label}: rodando… (pode levar alguns minutos; o treino já acabou)'

        def work():
            try:
                r = subprocess.run(cmd, cwd=core.ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                                   env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
                lines = [ln for ln in (r.stdout + '\n' + r.stderr).strip().splitlines() if ln.strip()]
                ok, tail = r.returncode == 0, (lines[-1] if lines else '')
            except OSError as e:
                ok, tail = False, str(e)
            self.busy = None
            if ok:
                self.say(f'{label}: pronto. {tail}', 3600, REWARD)
                self.ask = next_ask
            else:
                self.say(f'erro em {label}: {tail} (veja: make {"relatorio" if "relat" in label else "publicar"})', 3600)
        threading.Thread(target=work, daemon=True).start()

    def ask_number(self, label, current, lo, hi):
        """Digite o número na linha de baixo e Enter (Esc cancela)."""
        h, w = self.scr.getmaxyx()
        buf = ''
        self.scr.timeout(-1)
        try:
            while True:
                self.put(h - 2, 1, ' ' * (w - 3))
                self.put(h - 2, 1, f'{label} ({lo}-{hi}, atual {current}): {buf}█   Enter confirma · Esc cancela', INK, True)
                self.scr.refresh()
                k = self.scr.getch()
                if k in (10, 13, curses.KEY_ENTER):
                    break
                if k == 27:
                    return current
                if k in (curses.KEY_BACKSPACE, 127, 8):
                    buf = buf[:-1]
                elif 48 <= k <= 57 and len(buf) < 6:
                    buf += chr(k)
        finally:
            self.scr.timeout(250)
        if buf and lo <= int(buf) <= hi:
            self.say(f'{label}: {int(buf)} (vale no próximo treino)')
            return int(buf)
        self.say(f'erro: {label} deve ser um número entre {lo} e {hi}')
        return current

    def toggle(self):
        try:
            if self.S['running']:
                core.stop()
                self.say('parando: terminando a época e salvando best.pt/last.pt…', 600)
            elif not self.model:
                self.say('erro: nenhum .pt em models/ — coloque um modelo lá')
            else:
                core.start({'target': self.target, 'max_epochs': self.epochs, 'model': self.model,
                            'batch': BATCH_CHOICES[self.batch_i], 'ram_gb': RAM_CHOICES[self.ram_i],
                            'vram_gb': VRAM_CHOICES[self.vram_i], 'imgsz': IMGSZ_CHOICES[self.imgsz_i],
                            'patience': self.patience, 'lr0': LR_CHOICES[self.lr_i], 'freeze': FREEZE_CHOICES[self.freeze_i],
                            'amp': AMP_CHOICES[self.amp_i], 'cache': CACHE_CHOICES[self.cache_i],
                            'workers': WORKERS_CHOICES[self.workers_i]})
                self.msg = ''
        except ValueError as e:
            self.say(f'erro: {e}')

    def key(self, k):
        if self.ask and k not in (ord('q'), ord('Q')):   # pergunta pós-treino: só s/n (Esc = não)
            if k in (ord('s'), ord('S'), ord('y'), ord('Y')):
                self.answer(True)
            elif k in (ord('n'), ord('N'), 27):
                self.answer(False)
            return True
        if self.busy and k in (ord('s'), ord('S')):
            self.say('aguarde: ainda executando o passo anterior…')
            return True
        if k == ord('?'):
            self.show_help = not self.show_help
            return True
        if self.S['running'] and k in CONFIG_KEYS:
            self.say('erro: não dá pra mudar um treino em andamento — pare (s) e inicie de novo com a nova configuração')
            return True
        if k in (ord('q'), ord('Q')):
            if self.S['running'] and not self.quitting:
                core.stop()
                self.quitting = True
                self.say('parando: terminando a época e salvando best.pt/last.pt… (q de novo força a saída)', 600)
                return True
            core.kill()
            return False
        if k in (ord('s'), ord('S')):
            self.toggle()
        elif k in (ord('+'), ord('=')):
            self.target = min(0.99, round(self.target + 0.05, 2))
        elif k in (ord('-'), ord('_')):
            self.target = max(0.05, round(self.target - 0.05, 2))
        elif k in (ord('e'), ord('E')):
            self.epochs = self.ask_number('épocas', self.epochs, 1, 1000)
        elif k in (ord('m'), ord('M')):
            ms = self.S['models']
            if ms:
                self.model = ms[(ms.index(self.model) + 1) % len(ms)] if self.model in ms else ms[0]
        elif k in (ord('z'), ord('Z')):
            self.imgsz_i = (self.imgsz_i + 1) % len(IMGSZ_CHOICES)
        elif k in (ord('p'), ord('P')):
            self.patience = self.ask_number('patience', self.patience, 1, 1000)
        elif k in (ord('a'), ord('A')):
            self.lr_i = (self.lr_i + 1) % len(LR_CHOICES)
        elif k in (ord('f'), ord('F')):
            self.freeze_i = (self.freeze_i + 1) % len(FREEZE_CHOICES)
        elif k in (ord('b'), ord('B')):
            self.batch_i = (self.batch_i + 1) % len(BATCH_CHOICES)
        elif k in (ord('r'), ord('R')):
            self.ram_i = (self.ram_i + 1) % len(RAM_CHOICES)
        elif k in (ord('v'), ord('V')):
            self.vram_i = (self.vram_i + 1) % len(VRAM_CHOICES)
        elif k in (ord('u'), ord('U')):
            self.amp_i = (self.amp_i + 1) % len(AMP_CHOICES)
        elif k in (ord('c'), ord('C')):
            self.cache_i = (self.cache_i + 1) % len(CACHE_CHOICES)
        elif k in (ord('w'), ord('W')):
            self.workers_i = (self.workers_i + 1) % len(WORKERS_CHOICES)
        elif k in (ord('o'), ord('O')):
            if core.REPORT.exists():
                import webbrowser
                if not webbrowser.open(core.REPORT.as_uri()):
                    self.say(f'erro: sem navegador; abra {core.REPORT}')
            else:
                self.say('erro: ainda não há relatório (termine um treino ou rode make relatorio)')
        elif k in (ord('i'), ord('I')):
            g = core.write_gallery()
            if not g:
                self.say('erro: ainda não há imagens (elas surgem na 1ª época do treino)')
            else:
                self.say(f'imagens copiadas em {g} (abra a pasta no VS Code)')
        elif k in (ord('l'), ord('L')):
            self.show_log = not self.show_log
        elif k == ord('t'):
            self.show_sys = not self.show_sys
        elif k == ord('T'):
            self.run_btop()
        return True

    def check_thermal(self):
        """Se a GPU ficar >= MAX_TEMP em leituras seguidas, pede a parada graciosa (termina a época e salva best.pt/last.pt)."""
        if not MAX_TEMP or not self.S['running']:
            self.hot, self.hot_seen, self.thermal_stop = 0, None, False
            return
        d = self.mon.snapshot()['data']
        g = d.get('gpu')
        gt, ct = g and g.get('temp'), d.get('cpu_temp')
        if (gt is None and ct is None) or self.thermal_stop:
            return
        # a cada 2s: conta uma leitura quente se GPU >= MAX_TEMP ou CPU >= MAX_CPU_TEMP
        now = time.time()
        if self.hot_seen is None or now - self.hot_seen >= 2.0:
            self.hot_seen = now
            hot_g = gt is not None and gt >= MAX_TEMP
            hot_c = bool(MAX_CPU_TEMP) and ct is not None and ct >= MAX_CPU_TEMP
            self.hot = self.hot + 1 if (hot_g or hot_c) else 0
            if hot_g:
                self.hot_what = f'GPU em {gt:.0f}°C (limite {MAX_TEMP}°C)'
            elif hot_c:
                self.hot_what = f'CPU em {ct:.0f}°C (limite {MAX_CPU_TEMP}°C)'
        if self.hot >= HOT_SAMPLES:
            self.thermal_stop = True
            core.stop()
            self.say(f'🔥 {self.hot_what}: parando o treino — termina a época e salva best.pt/last.pt. '
                     f'Deixe esfriar e inicie de novo (s).', 3600)

    def run_btop(self):
        """Pausa a TUI e abre o btop de verdade; o treino segue rodando em segundo plano. Sai do btop com q."""
        if not shutil.which('btop'):
            self.say('erro: btop não instalado — sudo apt install btop (a tecla t já mostra CPU/RAM/GPU/temperaturas)')
            return
        curses.def_prog_mode()
        curses.endwin()
        try:
            subprocess.run(['btop'])
        finally:
            curses.reset_prog_mode()
            self.scr.clear()
            self.scr.refresh()

    # ---------- desenho ----------
    def draw(self):
        scr, S = self.scr, self.S
        scr.erase()
        self.pairs, self.next_pair = {}, 8   # pares de cor são recriados a cada quadro (sem vazar)
        h, w = scr.getmaxyx()
        if h < 24 or w < 80:
            self.put(0, 0, f'Terminal pequeno ({w}x{h}). Precisa de pelo menos 80x24.', PUNISH)
            return
        ev = S['events']
        eps = [e for e in ev if e['type'] == 'epoch']
        start = next((e for e in ev if e['type'] == 'start'), None)
        last = eps[-1] if eps else None
        goal = any(e['type'] == 'goal' for e in ev)
        target = start['target'] if start else self.target
        total = (last or start or {}).get('epochs', self.epochs)

        # cabeçalho
        led, lc = ('●', REWARD) if S['running'] else (('●', PUNISH) if (S['exit_code'] or 0) > 0 and not goal else ('●', DIM))
        self.put(0, 1, 'SENSUS·LAB', INK, True)
        self.put(0, 12, 'treino supervisionado', DIM)
        if S['running']:
            st = (f'treinando · época {last["epoch"]}/{total} · {int(last["t"] - (start or last)["t"])}s' if last
                  else ('carregando modelo…' if S['dataset_ready'] else 'baixando e balanceando o COCO…'))
        else:
            st = 'alvo atingido' if goal else 'treino encerrado' if last else 'pronto'
        self.put(0, max(36, w - len(st) - 4), led, lc, True)
        self.put(0, max(38, w - len(st) - 2), st, INK)
        self.rule(1)
        if self.show_log:
            return self.log_view()
        if self.show_help:
            return self.help_view()
        if self.show_sys:
            return self.sys_view()

        # herói: dígitos grandes + veredito
        base = (start or {}).get('baseline') or 0.0
        m = last['map50'] if last else base
        v = last.get('verdict') if last else None
        dig_pair = INK if last else DIM
        for i, row in enumerate(digits_rows(m)):
            self.put(2 + i, 2, row, dig_pair, True)
        self.put(2, 33, 'mAP50 · quanto a rede acerta', DIM)
        if v and last.get('prev_map50') is not None:
            d = last['map50'] - last['prev_map50']
            col = REWARD if v == 'reward' else PUNISH
            self.put(3, 33, ('▲ RECOMPENSA' if v == 'reward' else '▼ PUNIÇÃO'), col, True)
            self.put(4, 33, f'{d:+.3f} mAP50', col)
            self.put(5, 33, 'errou menos: pesos reforçados' if v == 'reward' else 'errou mais: gradiente corrige', DIM)
        else:
            self.put(3, 33, '—', DIM)
            self.put(5, 33, ('ponto de partida · melhorando o best.pt' if base else 'aguardando 1ª época') if not last else 'sem variação', DIM)
        nr, np_ = sum(e.get('verdict') == 'reward' for e in eps), sum(e.get('verdict') == 'punish' for e in eps)
        self.put(6, 33, f'▲ {nr}', REWARD, True)
        self.put(6, 41, f'▼ {np_}', PUNISH, True)

        ram, vram = RAM_CHOICES[self.ram_i], VRAM_CHOICES[self.vram_i]
        bs = BATCH_CHOICES[self.batch_i]
        cfg = (f'épocas {self.epochs} · batch {bs or "máx"} · '
               f'RAM {f"{ram}GB" if ram else "máx"} · VRAM {f"{vram}GB" if vram else "máx"} · '
               f'modelo {(self.model.split("/")[-1][:-3] or "—") if self.model else "—"}')
        self.put(7, 2, cfg, DIM)
        lr, fr = LR_CHOICES[self.lr_i], FREEZE_CHOICES[self.freeze_i]
        self.put(8, 2, f'imgsz {IMGSZ_CHOICES[self.imgsz_i]} · patience {self.patience} · LR {lr if lr else "auto"} · freeze {fr if fr >= 0 else "auto"}', DIM)
        wk = WORKERS_CHOICES[self.workers_i]
        self.put(9, 2, f'AMP {AMP_CHOICES[self.amp_i]} · cache {CACHE_CHOICES[self.cache_i].replace("false", "off")} · workers {wk or "máx"}', DIM)

        # régua até o alvo
        bw = min(w - 6, 70)
        filled = int(m * bw)
        tpos = int(target * bw)
        self.put(10, 2, '█' * filled + '─' * (bw - filled), INK if last else DIM)
        self.put(10, 2 + tpos, '┃', REWARD, True)
        self.put(11, 2, '0', DIM)
        self.put(11, 2 + bw - 1, '1', DIM)
        self.put(11, max(2, 2 + tpos - 3), f'ALVO {target:.2f}', REWARD)
        y = 12

        # gráfico de barras: mAP50 (verde=recompensa, vermelho=punição) + sparklines de loss
        rows = max(4, min(8, h - y - 17))
        self.rule(y, 'mAP50 por época')
        y += 1
        left = 5
        plot_w = w - left - 2
        cw = max(1, min(4, plot_w // max(total or 1, len(eps), 20)))
        shown = eps[-(plot_w // cw):] if eps else []
        for r in range(rows):
            lab = '1.0' if r == 0 else ('0.5' if r == rows // 2 else ('0.0' if r == rows - 1 else ''))
            self.put(y + r, 0, f'{lab:>3}', DIM)
        trow = y + int((1 - target) * (rows - 1) + 0.5)
        self.put(trow, left, '┄' * plot_w, REWARD)
        for i, e in enumerate(shown):
            cells = bar_cells(e['map50'], rows)
            col = REWARD if e.get('verdict') == 'reward' else PUNISH if e.get('verdict') == 'punish' else INK
            for r in range(rows):
                ch = cells[rows - 1 - r]
                if ch != ' ':
                    self.put(y + r, left + i * cw, ch * cw, col)
        y += rows
        if not eps:
            self.put(y - rows // 2, left + 4, 'aguardando a primeira época…', DIM)
        for name, key, pair in (('box', 'box_loss', L1), ('cls', 'cls_loss', L2), ('dfl', 'dfl_loss', L3)):
            if y >= h - 8:
                break
            vals = [(e.get('loss') or {}).get(key, 0) for e in eps]
            self.put(y, 0, f'{name}', pair)
            sp = ''.join(c * cw for c in spark(vals, (plot_w - 8) // cw)) if vals else ''
            self.put(y, left, sp, pair)
            if vals:
                self.put(y, left + len(sp) + 1, f'{vals[-1]:.2f}', pair)
            y += 1

        # neurônios + diário
        if y < h - 7:
            self.rule(y, 'neurônios · camada 1 · barra = quanto mudou')
            y += 1
            f = (last or start or {}).get('filters')
            if f:
                self.draw_neurons(f, y, h, w)
            else:
                self.put(y, 2, '(sem dados dos filtros)', DIM)
            if w >= 100 and eps:  # diário à direita
                dx = 76
                self.put(y - 1, dx - 2, '', DIM)
                self.put(y, dx, ' ép   mAP50   box    Δw%  veredito', DIM)
                for i, e in enumerate(reversed(eps[-6:])):
                    vd = e.get('verdict')
                    self.put(y + 1 + i, dx, f'{e["epoch"]:>3}  {e["map50"]:.3f}  {(e.get("loss") or {}).get("box_loss", 0):.2f}  '
                             f'{(e.get("delta_pct") or 0):>5.2f}  ', INK)
                    self.put(y + 1 + i, dx + 28, '▲' if vd == 'reward' else '▼' if vd == 'punish' else '·',
                             REWARD if vd == 'reward' else PUNISH if vd == 'punish' else DIM)

        # banner + log + rodapé
        end = next((e for e in ev if e['type'] == 'end'), None)
        ban, bp = '', DIM
        if (S['exit_code'] or 0) > 0 and not goal and not end:
            ban, bp = f'FALHOU (código {S["exit_code"]}): ' + ' · '.join(S['log'][-2:]), PUNISH
        elif end:
            ban, bp = end['text'], (REWARD if end['reason'] == 'goal' else DIM)
        elif goal:
            ban, bp = f'ALVO ATINGIDO: mAP50 {last["map50"]:.3f} ≥ {target:.2f}. Modelo em models/best.pt', REWARD
        elif S['running'] and not last:
            ban = 'Preparando… (pressione l para ver a saída do processo)'
        elif not S['running'] and not last and not S['dataset_ready']:
            ban = 'Dataset COCO não encontrado: será baixado e balanceado antes do treino.'
        elif not S['running'] and not last:
            ban = (f'Para quando: acabarem {self.epochs} épocas · ou mAP50 ≥ {self.target:.2f} · '
                   f'ou {self.patience} épocas sem melhorar  (? explica tudo)')
        if S.get('report_ready') and not S['running'] and not ban:
            ban = 'Relatório com matriz de confusão pronto: tecla o abre no navegador (runs/report/latest.html)'
        if self.msg and time.time() < self.msg_until:
            ban, bp = self.msg, self.msg_pair
        if self.busy:
            ban, bp = self.busy, L1
        elif self.ask:
            ban, bp = self.PROMPTS[self.ask], REWARD
        files = S.get('files') or []
        if files and not S['running']:
            self.put(h - 3, 1, ('Arquivos do run: ' + ' · '.join(f'{f["name"]} {f["size"] / 1e6:.1f}MB' if f['size'] > 1e6 else f'{f["name"]}' for f in files)
                                + f'  em {S.get("run_dir", "")}')[: w - 3], DIM)
        else:
            self.sys_strip(h - 3)
        self.footer(ban, bp)

    def sys_strip(self, y):
        """Uma linha com os sensores principais (a tecla t abre o painel completo)."""
        d = self.mon.snapshot()['data']
        if not d:
            return
        g, x = d.get('gpu'), 1
        segs = [('CPU ', DIM), (f'{d["cpu"]:.0f}%' if d.get('cpu') is not None else '--', INK)]
        if d.get('cpu_temp') is not None:
            segs += [(f' {d["cpu_temp"]:.0f}°C', temp_pair(d['cpu_temp']))]
        if d.get('ram_total'):
            segs += [('   RAM ', DIM), (f'{d["ram_used"]:.1f}/{d["ram_total"]:.0f}GB', INK)]
        if g:
            segs += [('   GPU ', DIM), (f'{g["util"]:.0f}%' if g['util'] is not None else '--', INK)]
            if g['temp'] is not None:
                segs += [(f' {g["temp"]:.0f}°C', temp_pair(g['temp']))]
            if g['mem_total']:
                segs += [('   VRAM ', DIM), (f'{g["mem_used"] / 1024:.1f}/{g["mem_total"] / 1024:.0f}GB', INK)]
        segs += [(f'   (t = sensores · para a {MAX_TEMP}°C)' if MAX_TEMP else '   (t = sensores)', DIM)]
        for text, pair in segs:
            self.put(y, x, text, pair)
            x += len(text)

    def sys_view(self):
        """Painel de sensores estilo btop: barras + histórico de CPU, RAM, GPU, VRAM e temperaturas."""
        h, w = self.scr.getmaxyx()
        self.rule(1, 'sensores · estilo btop (t volta · T abre o btop de verdade)')
        s = self.mon.snapshot()
        d, hist = s['data'], s['hist']
        if not d:
            self.put(3, 2, 'lendo sensores…', DIM)
            return self.footer('', DIM)
        bw = max(10, min(w - 44, 48))
        sx = 12 + bw + 16            # coluna do histórico
        sw = w - sx - 2

        def row(y, label, frac, text, pair, key=None, top=100.0):
            self.put(y, 2, f'{label:<6}', INK, True)
            n = int(max(0.0, min(1.0, frac)) * bw)
            self.put(y, 9, '█' * n, pair)
            self.put(y, 9 + n, '─' * (bw - n), DIM)
            self.put(y, 10 + bw, text, INK)
            if key and sw >= 10 and hist.get(key):
                self.put(y, sx, spark_abs(hist[key], sw, top), pair)

        y = 3
        cpu = d.get('cpu')
        row(y, 'CPU', (cpu or 0) / 100, f'{cpu:5.1f}%' if cpu is not None else '  n/d ', L2, 'cpu')
        y += 1
        ct = d.get('cpu_temp')
        if ct is not None:
            src = d.get('cpu_temp_src')
            row(y, 'temp', ct / 100, f'{ct:5.1f}°C' + (f' · via {src}' if src and src != 'linux' else ''), temp_pair(ct), 'ctemp')
        else:
            self.put(y, 2, 'temp  ', INK, True)
            self.put(y, 9, 'n/d — no Windows abra o LibreHardwareMonitor (admin) > Options > Remote Web Server'[: w - 11], DIM)
        y += 1
        if d.get('cores'):
            self.put(y, 2, 'cores ', INK, True)
            self.put(y, 9, ''.join(EIGHTHS[max(1, min(8, round(8 * c / 100)))] for c in d['cores'])[: w - 12], L2)
            y += 1
        y += 1
        if d.get('ram_total'):
            frac = d['ram_used'] / d['ram_total']
            row(y, 'RAM', frac, f'{d["ram_used"]:4.1f}/{d["ram_total"]:.0f}GB', L1, 'ram')
        y += 2
        g = d.get('gpu')
        if not g:
            self.put(y, 2, 'GPU   ', INK, True)
            self.put(y, 9, 'nvidia-smi não encontrado (sem GPU NVIDIA visível)', DIM)
        else:
            self.put(y, 2, g['name'], REWARD, True)
            y += 1
            u = g['util']
            row(y, 'GPU', (u or 0) / 100, f'{u:5.1f}%' if u is not None else '  n/d ', REWARD, 'gpu')
            y += 1
            if g['mem_total']:
                row(y, 'VRAM', g['mem_used'] / g['mem_total'], f'{g["mem_used"]:.0f}/{g["mem_total"]:.0f}MB', L3, 'vram')
                y += 1
            t = g['temp']
            if t is not None:
                row(y, 'temp', t / 100, f'{t:5.1f}°C', temp_pair(t), 'gtemp')
                y += 1
            extra = []
            if g['power'] is not None:
                extra.append(f'{g["power"]:.0f}' + (f'/{g["power_limit"]:.0f}' if g['power_limit'] else '') + ' W')
            if g['clock'] is not None:
                extra.append(f'{g["clock"]:.0f} MHz')
            if extra:
                self.put(y, 9, ' · '.join(extra), DIM)
                y += 1
        y += 1
        legenda = 'temperatura: verde < 70°C · âmbar < 80°C · vermelho ≥ 80°C (GPU esquenta mais no treino com batch máx)'
        if y < h - 3:
            self.put(y, 2, legenda[: w - 4], DIM)
        self.footer('', DIM)

    def footer(self, ban='', bp=DIM):
        h, w = self.scr.getmaxyx()
        self.put(h - 2, 1, ban[: w - 3], bp)
        info = (f'? ajuda · s {"parar" if self.S["running"] else "iniciar"} · +/- alvo {self.target:.2f} · e épocas · z imgsz · p patience · a LR · f freeze · b batch · r RAM · '
                f'v VRAM · u AMP · c cache · w workers · m modelo · i imagens · o relatório · t sensores · T btop · l {"voltar" if self.show_log else "log"} · q sair')
        self.put(h - 1, 1, info, DIM)

    def help_view(self):
        h, w = self.scr.getmaxyx()
        self.rule(1, 'ajuda · o que é cada configuração (? volta)')
        y = 3
        for title, text in HELP:
            self.put(y, 2, title, REWARD, True)
            y += 1
            for ln in textwrap.wrap(text, max(30, min(w - 6, 100))):
                if y < h - 3:
                    self.put(y, 4, ln, INK)
                y += 1
            y += 1
        self.footer('', DIM)

    def log_view(self):
        h, w = self.scr.getmaxyx()
        self.rule(1, 'saída do processo')
        lines = self.S['log'][-(h - 5):] or ['(sem saída ainda — inicie o treino com s)']
        for i, ln in enumerate(lines):
            self.put(2 + i, 1, ln[: w - 2], DIM)
        self.footer(self.msg, PUNISH)

    def run(self):
        while True:
            now = time.time()
            if now - self.last_poll > 1.0:
                self.S, self.last_poll = core.state(), now
                if self.quitting and not self.S['running']:
                    return
                if self.was_running and not self.S['running']:
                    self.after_training()
                self.was_running = self.S['running']
                if self.model not in self.S['models']:
                    self.model = self.S['models'][0] if self.S['models'] else ''
                if self.msg and time.time() > self.msg_until:
                    self.msg = ''
            self.check_thermal()
            self.draw()
            self.scr.refresh()
            k = self.scr.getch()
            if k == curses.KEY_RESIZE:
                self.scr.clear()
            elif k != -1 and not self.key(k):
                return


def main():
    try:
        curses.wrapper(lambda scr: UI(scr).run())
    except KeyboardInterrupt:
        core.stop()


if __name__ == '__main__':
    main()
