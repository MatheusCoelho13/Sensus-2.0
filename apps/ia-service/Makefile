# Sensus IA - treino. (Targets antigos removidos; histórico no git: git show HEAD:apps/ia-service/Makefile)
PER_CLASS ?= 1000
VAL_PER_CLASS ?= 200
MODEL ?= models/yolov8n.pt
TARGET ?= 0.60
# Recursos: 0 = usar o MÁXIMO; qualquer valor > 0 vira limite.
# BATCH: imagens por passo (0 = ocupa 90% da VRAM) | RAM_GB: cache de imagens em RAM (0 = toda a RAM livre)
# VRAM_GB: limite de GPU (0 = a GPU inteira)       | WORKERS: leitura de dados (0 = todos os núcleos da CPU)
BATCH ?= 0
RAM_GB ?= 0
VRAM_GB ?= 0
WORKERS ?= 0
MEM = --batch $(BATCH) --ram-gb $(RAM_GB) --vram-gb $(VRAM_GB) --workers $(WORKERS)
PY = $(if $(wildcard .venv/bin/activate),. .venv/bin/activate && python3,python3)

.DEFAULT_GOAL := help
.PHONY: help tui painel dataset treinar train_coco coco relatorio publicar

help:
	@echo "Sensus IA - comandos:"
	@echo "  make tui         - interface no terminal (inicia/para o treino, gráficos ao vivo)"
	@echo "  make painel      - interface gráfica no navegador (http://127.0.0.1:8765)"
	@echo "  make dataset     - baixa o COCO (10 classes) e balanceia (PER_CLASS=1000 VAL_PER_CLASS=200)"
	@echo "  make treinar     - treina até TARGET de mAP50 (padrão 0.60); baixa o dataset se faltar"
	@echo "  make train_coco  - treino rápido (patience 15) com o dataset já pronto"
	@echo "  make coco        - dataset + train_coco"
	@echo "  make publicar    - sobe models/best.pt + relatório pro GitHub (commit só desses arquivos). Ex: make treinar publicar"
	@echo "  make relatorio   - relatório com matriz de confusão por classe (WEIGHTS=models/best.pt); também sai no fim do treino"
	@echo "  recursos (0 = máximo, padrão): BATCH= RAM_GB= VRAM_GB= WORKERS=   Ex: make treinar VRAM_GB=4 BATCH=8"

tui:
	$(PY) scripts/train_tui.py

painel:
	$(PY) scripts/train_ui.py

dataset:
	@echo "Baixando COCO (só as 10 classes) e balanceando: $(PER_CLASS) imgs/classe train, $(VAL_PER_CLASS) val"
	$(PY) scripts/build_coco_dataset.py --per-class $(PER_CLASS) --val-per-class $(VAL_PER_CLASS)

treinar:
	@[ -d datasets/coco_nav/images/train ] || $(MAKE) dataset
	$(PY) scripts/train.py --data config/data.yaml --model $(MODEL) --imgsz 640 $(MEM) \
		--cls-weight auto --max-epochs 300 --patience 100 --target-map $(TARGET)

train_coco:
	$(PY) scripts/train.py --data config/data.yaml --model $(MODEL) --imgsz 640 $(MEM) --cls-weight auto --patience 15

coco: dataset train_coco

WEIGHTS ?= models/best.pt
relatorio:
	$(PY) scripts/report.py --weights $(WEIGHTS) --data config/data.yaml --imgsz 640

publicar:
	$(PY) scripts/publish.py
