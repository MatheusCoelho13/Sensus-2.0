#!/usr/bin/env python3
"""Confere se o ambiente está pronto para treinar. Sai com código 1 se faltar algo obrigatório.

  make verificar
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = [  # (módulo, pacote pip)
    ('ultralytics', 'ultralytics'), ('torch', 'torch'), ('yaml', 'pyyaml'), ('matplotlib', 'matplotlib'),
    ('requests', 'requests'), ('tqdm', 'tqdm'), ('numpy', 'numpy'), ('PIL', 'Pillow'), ('cv2', 'opencv-python-headless'),
]


def check_modules():
    missing = []
    for mod, pip in REQUIRED:
        try:
            m = importlib.import_module(mod)
            print(f'  ✅ {mod:<12} {getattr(m, "__version__", "")}')
        except Exception:  # noqa: BLE001
            print(f'  ❌ {mod:<12} FALTA (pip install {pip})')
            missing.append(pip)
    return missing


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f'  ✅ GPU        {p.name} · {p.total_memory / 1e9:.1f} GB de VRAM  (o treino usa a GPU)')
        else:
            print('  ⚠️ GPU        nenhuma GPU CUDA: o treino vai rodar na CPU (mais lento). Normal se não tem placa NVIDIA.')
    except Exception:  # noqa: BLE001
        pass


def check_project():
    models = sorted(p.name for p in (ROOT / 'models').glob('*.pt'))
    print(f'  {"✅" if models else "⚠️"} modelos      ' + (', '.join(models) if models else 'nenhum .pt em models/ — coloque um (ex: yolov8n.pt) ou o ultralytics baixa'))
    cfg = ROOT / 'config' / 'data.yaml'
    try:
        import yaml
        path = Path(yaml.safe_load(cfg.read_text()).get('path', ''))
        ok = path.is_dir() and (path / 'images' / 'train').exists()
        print(f'  {"✅" if ok else "⚠️"} dataset      ' + (f'{path}' if ok else f'não encontrado ({path}) — rode: make dataset'))
    except Exception as e:  # noqa: BLE001
        print(f'  ⚠️ dataset      não consegui ler config/data.yaml ({type(e).__name__})')


def main():
    print(f'Python {sys.version.split()[0]} em {sys.prefix}')
    print('Dependências:')
    missing = check_modules()
    check_gpu()
    print('Projeto:')
    check_project()
    if missing:
        print(f'\n❌ Faltam {len(missing)} pacote(s). Rode: make instalar')
        sys.exit(1)
    print('\n✅ Tudo certo para treinar: make tui   (ou make treinar)')


if __name__ == '__main__':
    main()
