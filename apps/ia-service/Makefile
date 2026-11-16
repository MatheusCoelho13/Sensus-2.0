# Sensus IA - treino. (Targets antigos removidos; histórico no git: git show HEAD:apps/ia-service/Makefile)
PER_CLASS ?= 1000
VAL_PER_CLASS ?= 200
MODEL ?= models/yolov8n.pt
TARGET ?= 0.60
PY = $(if $(wildcard .venv/bin/activate),. .venv/bin/activate && python3,python3)

.DEFAULT_GOAL := help
.PHONY: help tui painel dataset treinar train_coco coco

help:
	@echo "Sensus IA - comandos:"
	@echo "  make tui         - interface no terminal (inicia/para o treino, gráficos ao vivo)"
	@echo "  make painel      - interface gráfica no navegador (http://127.0.0.1:8765)"
	@echo "  make dataset     - baixa o COCO (10 classes) e balanceia (PER_CLASS=1000 VAL_PER_CLASS=200)"
	@echo "  make treinar     - treina até TARGET de mAP50 (padrão 0.60); baixa o dataset se faltar"
	@echo "  make train_coco  - treino rápido (patience 15) com o dataset já pronto"
	@echo "  make coco        - dataset + train_coco"

tui:
	$(PY) scripts/train_tui.py

painel:
	$(PY) scripts/train_ui.py

dataset:
	@echo "Baixando COCO (só as 10 classes) e balanceando: $(PER_CLASS) imgs/classe train, $(VAL_PER_CLASS) val"
	$(PY) scripts/build_coco_dataset.py --per-class $(PER_CLASS) --val-per-class $(VAL_PER_CLASS)

treinar:
	@[ -d datasets/coco_nav/images/train ] || $(MAKE) dataset
	$(PY) scripts/train.py --data config/data.yaml --model $(MODEL) --imgsz 640 --batch 16 \
		--cls-weight auto --max-epochs 300 --patience 100 --target-map $(TARGET)

train_coco:
	$(PY) scripts/train.py --data config/data.yaml --model $(MODEL) --imgsz 640 --batch 16 --cls-weight auto --patience 15

coco: dataset train_coco
