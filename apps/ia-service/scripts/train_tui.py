#!/usr/bin/env python3
"""Painel do treino NO TERMINAL (curses, stdlib pura).

  make tui

Mesmo motor do painel web (scripts/train_ui.py): sobe o train.py, lê runs/ui/events.jsonl.
Teclas: s iniciar/parar · +/- alvo mAP50 · e épocas · b batch · r RAM · v VRAM · m modelo · l log · q sair (para o treino)
"""
import curses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
EPOCH_CHOICES = [50, 100, 300, 600]
BATCH_CHOICES = [4, 8, 16, 32, 64]
RAM_CHOICES = [0, 2, 4, 8, 16, 32]    # GB p/ cache de imagens; 0 = automático
VRAM_CHOICES = [0, 2, 3, 4, 6, 8, 12]  # GB de GPU; 0 = sem limite

INK, REWARD, PUNISH, DIM, L1, L2, L3 = range(1, 8)


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


class UI:
    def __init__(self, scr):
        self.scr, self.msg, self.show_log = scr, '', False
        self.target, self.epochs_i, self.model = 0.60, 2, ''
        self.batch_i, self.ram_i, self.vram_i = 2, 0, 0
        self.S, self.last_poll, self.pairs = {'running': False, 'events': [], 'log': [], 'dataset_ready': True, 'models': [],
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
        n, avail = len(tiles), h - 3 - y
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
            if ty + th + 1 > h - 2:
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
    def toggle(self):
        try:
            if self.S['running']:
                core.stop()
                self.msg = 'parando treino…'
            elif not self.model:
                self.msg = 'erro: nenhum .pt em models/ — coloque um modelo lá'
            else:
                core.start({'target': self.target, 'max_epochs': EPOCH_CHOICES[self.epochs_i], 'model': self.model,
                            'batch': BATCH_CHOICES[self.batch_i], 'ram_gb': RAM_CHOICES[self.ram_i],
                            'vram_gb': VRAM_CHOICES[self.vram_i]})
                self.msg = ''
        except ValueError as e:
            self.msg = f'erro: {e}'

    def key(self, k):
        if k in (ord('q'), ord('Q')):
            core.stop()
            return False
        if k in (ord('s'), ord('S')):
            self.toggle()
        elif k in (ord('+'), ord('=')):
            self.target = min(0.99, round(self.target + 0.05, 2))
        elif k in (ord('-'), ord('_')):
            self.target = max(0.05, round(self.target - 0.05, 2))
        elif k in (ord('e'), ord('E')):
            self.epochs_i = (self.epochs_i + 1) % len(EPOCH_CHOICES)
        elif k in (ord('m'), ord('M')):
            ms = self.S['models']
            if ms:
                self.model = ms[(ms.index(self.model) + 1) % len(ms)] if self.model in ms else ms[0]
        elif k in (ord('b'), ord('B')):
            self.batch_i = (self.batch_i + 1) % len(BATCH_CHOICES)
        elif k in (ord('r'), ord('R')):
            self.ram_i = (self.ram_i + 1) % len(RAM_CHOICES)
        elif k in (ord('v'), ord('V')):
            self.vram_i = (self.vram_i + 1) % len(VRAM_CHOICES)
        elif k in (ord('l'), ord('L')):
            self.show_log = not self.show_log
        return True

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
        total = (last or start or {}).get('epochs', EPOCH_CHOICES[self.epochs_i])

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

        # herói: dígitos grandes + veredito
        m = last['map50'] if last else 0.0
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
            self.put(5, 33, 'aguardando 1ª época' if not last else 'sem variação', DIM)
        nr, np_ = sum(e.get('verdict') == 'reward' for e in eps), sum(e.get('verdict') == 'punish' for e in eps)
        self.put(6, 33, f'▲ {nr}', REWARD, True)
        self.put(6, 41, f'▼ {np_}', PUNISH, True)

        ram, vram = RAM_CHOICES[self.ram_i], VRAM_CHOICES[self.vram_i]
        cfg = (f'épocas {EPOCH_CHOICES[self.epochs_i]} · batch {BATCH_CHOICES[self.batch_i]} · '
               f'RAM {f"{ram}GB" if ram else "auto"} · VRAM {f"{vram}GB" if vram else "livre"} · '
               f'modelo {(self.model.split("/")[-1][:-3] or "—") if self.model else "—"}')
        self.put(7, 2, cfg, DIM)

        # régua até o alvo
        bw = min(w - 6, 70)
        filled = int(m * bw)
        tpos = int(target * bw)
        self.put(8, 2, '█' * filled + '─' * (bw - filled), INK if last else DIM)
        self.put(8, 2 + tpos, '┃', REWARD, True)
        self.put(9, 2, '0', DIM)
        self.put(9, 2 + bw - 1, '1', DIM)
        self.put(9, max(2, 2 + tpos - 3), f'ALVO {target:.2f}', REWARD)
        y = 10

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
        ban, bp = '', DIM
        if (S['exit_code'] or 0) > 0 and not goal:
            ban, bp = f'FALHOU (código {S["exit_code"]}): ' + ' · '.join(S['log'][-2:]), PUNISH
        elif goal:
            ban, bp = f'ALVO ATINGIDO: mAP50 {last["map50"]:.3f} ≥ {target:.2f}. Modelo em models/best.pt', REWARD
        elif S['running'] and not last:
            ban = 'Preparando… (pressione l para ver a saída do processo)'
        elif not S['running'] and not last and not S['dataset_ready']:
            ban = 'Dataset COCO não encontrado: será baixado e balanceado antes do treino.'
        if self.msg:
            ban, bp = self.msg, PUNISH
        self.footer(ban, bp)

    def footer(self, ban='', bp=DIM):
        h, w = self.scr.getmaxyx()
        self.put(h - 2, 1, ban[: w - 3], bp)
        info = (f's {"parar" if self.S["running"] else "iniciar"} · +/- alvo {self.target:.2f} · e épocas · b batch · r RAM · '
                f'v VRAM · m modelo · l {"voltar" if self.show_log else "log"} · q sair')
        self.put(h - 1, 1, info, DIM)

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
                if self.model not in self.S['models']:
                    self.model = self.S['models'][0] if self.S['models'] else ''
                if self.S['running'] and self.msg.startswith('parando'):
                    pass
                elif not self.S['running']:
                    self.msg = self.msg if self.msg.startswith('erro') else ''
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
