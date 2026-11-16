#!/usr/bin/env python3
"""Sobe o resultado do treino para o GitHub: models/best.pt + o último relatório.

  make publicar                 # commit + push na branch atual
  make treinar publicar         # treina e, se terminar bem, publica
  python3 scripts/publish.py --dry-run

Só entram no commit: models/best.pt e reports/<data-hora>/relatorio.{html,json} (copiados de
runs/report/). Qualquer outra mudança do seu working tree fica de fora. O *.pt é ignorado pelo
.gitignore da raiz, então o add é forçado (-f) apenas nesses arquivos.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

MAX_MB = 90  # GitHub rejeita arquivos > 100MB; acima disso use `gh release` em vez de commit


def git(root, *args, check=True):
    r = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f'❌ git {" ".join(args)}:\n{(r.stderr or r.stdout).strip()}')
    return r


def latest_report(root: Path):
    dirs = sorted(d for d in (root / 'runs' / 'report').glob('*') if d.is_dir() and (d / 'relatorio.html').exists())
    return dirs[-1] if dirs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=str(Path(__file__).resolve().parent.parent), help='pasta do ia-service')
    ap.add_argument('--remote', default='origin')
    ap.add_argument('--dry-run', action='store_true', help='mostra o que faria, sem commitar nem enviar')
    a = ap.parse_args()
    root = Path(a.root).resolve()

    best = root / 'models' / 'best.pt'
    if not best.exists():
        raise SystemExit('❌ models/best.pt não existe — treine primeiro (make treinar)')
    mb = best.stat().st_size / 1e6
    if mb > MAX_MB:
        raise SystemExit(f'❌ best.pt tem {mb:.0f}MB (> {MAX_MB}MB): o GitHub recusa. Use: gh release create modelo-v1 {best}')

    rep = latest_report(root)
    files = [best]
    map50 = None
    if rep:
        dest = root / 'reports' / rep.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in ('relatorio.html', 'relatorio.json'):
            if (rep / f).exists():
                shutil.copy2(rep / f, dest / f)
                files.append(dest / f)
        try:
            map50 = json.loads((rep / 'relatorio.json').read_text())['summary']['map50']
        except (OSError, KeyError, ValueError):
            pass
    else:
        print('⚠️ nenhum relatório em runs/report/ — subindo só o modelo (gere com: make relatorio)')

    branch = git(root, 'rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()
    rel = [str(f.relative_to(git(root, 'rev-parse', '--show-toplevel').stdout.strip())) for f in files]
    msg = f'chore(ia-service): modelo treinado' + (f' mAP50={map50:.3f}' if map50 is not None else '') + \
          (f' ({rep.name})' if rep else '')
    print(f'📦 {len(files)} arquivo(s) → {a.remote}/{branch}\n   ' + '\n   '.join(rel) + f'\n   "{msg}"')
    if a.dry_run:
        print('🔎 dry-run: nada foi commitado nem enviado')
        return

    git(root, 'add', '-f', '--', *map(str, files))
    if not git(root, 'diff', '--cached', '--quiet', '--', *map(str, files), check=False).returncode:
        print('✅ nada novo para publicar (o modelo e o relatório já estão no repositório)')
        return
    git(root, 'commit', '-m', msg, '--', *map(str, files))   # só esses caminhos; o resto do staging não entra
    git(root, 'pull', '--rebase', '--autostash', a.remote, branch, check=False)
    r = git(root, 'push', a.remote, 'HEAD', check=False)
    if r.returncode:
        sys.exit(f'❌ push falhou:\n{r.stderr.strip()}\n💡 403? confira a conta: gh auth status (ou gh auth setup-git)\n'
                 '   O commit ficou local; depois de resolver rode: git push')
    print('🚀 publicado!', git(root, 'log', '-1', '--format=%h %s').stdout.strip())


if __name__ == '__main__':
    main()
