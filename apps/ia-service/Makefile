# Sensus IA - treino. (Targets antigos removidos; histórico no git: git show HEAD:apps/ia-service/Makefile)
PER_CLASS ?= 1000
VAL_PER_CLASS ?= 200
# Se existir models/best.pt o treino continua dele (refinando); senão parte do yolov8n.
MODEL ?= $(if $(wildcard models/best.pt),models/best.pt,models/yolov8n.pt)
# Parada: o treino termina quando acabam EPOCHS, OU o mAP50 chega a TARGET, OU passam PATIENCE épocas sem melhorar.
EPOCHS ?= 300
PATIENCE ?= 30
# Resolução de treino (px). Maior ajuda objetos pequenos (book, bottle, cell phone) mas gasta mais VRAM/tempo: 640 (padrão), 800, 960, 1280.
IMGSZ ?= 640
TARGET ?= 0.60
# Recursos: 0 = usar o MÁXIMO; qualquer valor > 0 vira limite.
# BATCH: imagens por passo (0 = ocupa 90% da VRAM) | RAM_GB: cache de imagens em RAM (0 = toda a RAM livre)
# VRAM_GB: limite de GPU (0 = a GPU inteira)       | WORKERS: leitura de dados (0 = todos os núcleos da CPU)
BATCH ?= 0
RAM_GB ?= 0
VRAM_GB ?= 0
WORKERS ?= 0
# DEVICE: auto | cpu | 0 | 0,1 | all  (várias GPUs = DDP; nesse modo painel/class-weights/alvo mAP não rodam, veja train.py)
DEVICE ?= auto
# Melhoria do dataset: TARGET_INST = instâncias por classe a atingir (0 = mediana das classes)
TARGET_INST ?= 0
# TUI: se a GPU chegar a MAX_TEMP °C (2 leituras seguidas) o treino é parado com segurança (salva best.pt/last.pt). 0 = desliga.
MAX_TEMP ?= 85
export MAX_TEMP
MEM = --batch $(BATCH) --ram-gb $(RAM_GB) --vram-gb $(VRAM_GB) --workers $(WORKERS) --device $(DEVICE)
PY = $(if $(wildcard .venv/bin/activate),. .venv/bin/activate && python3,python3)

.DEFAULT_GOAL := help
.PHONY: help instalar verificar tui painel dataset auditar melhorar melhorar_limpar treinar train_coco coco relatorio publicar

help:
	@echo "Sensus IA - comandos:"
	@echo "  make instalar    - cria o venv e instala tudo que precisa (make instalar CPU=1 = torch leve, só CPU) e confere no fim"
	@echo "  make verificar   - confere se o ambiente está pronto (pacotes, GPU, modelos, dataset)"
	@echo "  make tui         - interface no terminal (inicia/para o treino, gráficos ao vivo)"
	@echo "  make painel      - interface gráfica no navegador (http://127.0.0.1:8765)"
	@echo "  make dataset     - baixa o COCO (10 classes) e balanceia (PER_CLASS=1000 VAL_PER_CLASS=200)"
	@echo "  make auditar     - audita o dataset (corrompidas, duplicadas, vazamento train/val, balanço); não altera nada"
	@echo "  make melhorar    - corrige o dataset (irrecuperável vai p/ _quarantine/) e balanceia classes raras com flip/rotação/luz/ruído. Ex: make melhorar TARGET_INST=1500"
	@echo "  make melhorar_limpar - remove as imagens geradas por 'make melhorar'"
	@echo "  make treinar     - treina até TARGET de mAP50 (padrão 0.60); baixa o dataset se faltar. Várias GPUs: make treinar DEVICE=0,1 (ou all)"
	@echo "  make train_coco  - treino rápido (patience 15) com o dataset já pronto"
	@echo "  make coco        - dataset + train_coco"
	@echo "  make publicar    - sobe models/best.pt + relatório pro GitHub (commit só desses arquivos). Ex: make treinar publicar"
	@echo "  make relatorio   - relatório com matriz de confusão por classe (WEIGHTS=models/best.pt); também sai no fim do treino"
	@echo "  resolução: IMGSZ=640 (padrão; 960 p/ objetos pequenos, usa mais VRAM)  Ex: make treinar IMGSZ=960 BATCH=4"
	@echo "  parada: EPOCHS=300 PATIENCE=30 TARGET=0.60  (para quando acabam as épocas, ou mAP50 >= TARGET, ou PATIENCE épocas sem melhorar)"
	@echo "  segurança: a TUI para o treino se a GPU chegar a MAX_TEMP=85 °C (0 desliga). Ex: make tui MAX_TEMP=80"
	@echo "  recursos (0 = máximo, padrão): BATCH= RAM_GB= VRAM_GB= WORKERS=   Ex: make treinar VRAM_GB=4 BATCH=8"

instalar:
	CPU=$(CPU) bash scripts/instalar.sh

verificar:
	$(PY) scripts/verify_env.py

tui:
	$(PY) scripts/train_tui.py

painel:
	$(PY) scripts/train_ui.py

dataset:
	@echo "Baixando COCO (só as 10 classes) e balanceando: $(PER_CLASS) imgs/classe train, $(VAL_PER_CLASS) val"
	$(PY) scripts/build_coco_dataset.py --per-class $(PER_CLASS) --val-per-class $(VAL_PER_CLASS)

auditar:
	$(PY) scripts/improve_dataset.py --data config/data.yaml

melhorar:
	$(PY) scripts/improve_dataset.py --data config/data.yaml --fix --balance --target $(TARGET_INST)

melhorar_limpar:
	$(PY) scripts/improve_dataset.py --data config/data.yaml --clean

treinar:
	@[ -d datasets/coco_nav/images/train ] || $(MAKE) dataset
	$(PY) scripts/train.py --data config/data.yaml --model $(MODEL) --imgsz $(IMGSZ) $(MEM) \
		--cls-weight auto --max-epochs $(EPOCHS) --patience $(PATIENCE) --target-map $(TARGET)

train_coco:
	$(PY) scripts/train.py --data config/data.yaml --model $(MODEL) --imgsz $(IMGSZ) $(MEM) --cls-weight auto --max-epochs $(EPOCHS) --patience 15

coco: dataset train_coco

WEIGHTS ?= models/best.pt
relatorio:
	$(PY) scripts/report.py --weights $(WEIGHTS) --data config/data.yaml --imgsz $(IMGSZ)

publicar:
	$(PY) scripts/publish.py
