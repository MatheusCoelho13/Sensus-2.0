#!/usr/bin/env bash
# Instala tudo que o projeto precisa: venv + dependências + checagem final.
#
#   make instalar            # GPU NVIDIA ou CPU: no Linux o pip já traz o torch certo (~2GB)
#   make instalar CPU=1      # só CPU: torch leve (~200MB)
#   make verificar           # só confere se está tudo funcionando
#
# Variáveis (opcionais): PYTHON=python3.11  VENV=.venv  REQ=config/requirements.txt  DRY=1 (só mostra os passos)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}; VENV=${VENV:-.venv}; REQ=${REQ:-config/requirements.txt}
CPU=${CPU:-0}; DRY=${DRY:-0}; VERIFY=${VERIFY:-1}

run() { echo "  \$ $*"; [ "$DRY" = 1 ] || "$@"; }
fail() { echo "❌ $*" >&2; exit 1; }

echo "📦 Instalando o ambiente do Sensus IA ($([ "$CPU" = 1 ] && echo 'só CPU' || echo 'GPU/CPU automático'))"

command -v "$PYTHON" >/dev/null || fail "$PYTHON não encontrado. Instale o Python 3.9+ (ex: sudo apt install python3 python3-venv)"
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 9))' || fail "Python $("$PYTHON" -V 2>&1) é antigo: precisa de 3.9 ou mais novo"
echo "✓ $("$PYTHON" -V 2>&1)"

[ -f "$REQ" ] || fail "arquivo de dependências não encontrado: $REQ"

if [ -f "$VENV/bin/activate" ]; then
    echo "✓ venv $VENV já existe (reaproveitando)"
else
    echo "▶ criando venv em $VENV"
    run "$PYTHON" -m venv "$VENV" || fail "não consegui criar o venv. No Debian/Ubuntu: sudo apt install python3-venv"
fi

PIP=("$VENV/bin/python" -m pip)
echo "▶ atualizando o pip"
run "${PIP[@]}" install -q -U pip

if [ "$CPU" = 1 ]; then
    echo "▶ instalando torch só-CPU (mais leve)"
    run "${PIP[@]}" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "▶ instalando dependências ($REQ) — o torch é o maior, pode demorar"
run "${PIP[@]}" install -r "$REQ"

mkdir -p models runs
if [ "$VERIFY" = 1 ] && [ "$DRY" != 1 ]; then
    echo "▶ conferindo"
    "$VENV/bin/python" scripts/verify_env.py
else
    echo "✅ pronto (sem verificação)."
fi
