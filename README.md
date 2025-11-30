## Image Deblurring

---

### Описание

---

Проект для восстановления резкости изображений (image deblurring) с использованием стека **PyTorch Lightning + Hydra**.

Основные компоненты:

- **PyTorch / Lightning** — обучение и логика моделей;
- **Hydra** — управление конфигами и экспериментами;
- **MLflow** — логирование экспериментов;

---

### Структура проекта

---

- **`src/`** — python‑пакет с основным кодом:
  - **`data/`** — датасеты и аугментации (albumentations);
  - **`models/`** — модели деблюринга
  - **`training/`** — Lightning‑модуль (`LitDeblurring`) и DataModule;
  - **`utils/`** — метрики, лоссы;
  - **`scripts/`** — вспомогательные скрипты для инференса.
- **`configs/`** — конфиги **Hydra** (данные, модель, обучение, логирование).
- **`data/preprocessing/`** — скрипты подготовки датасета (распаковка RSBlur, генерация `.npy`).
- **`train.py`** — входная точка обучения (Hydra + PyTorch Lightning).

---

### Setup (uv + editable install)

---

#### 1. Клонировать репозиторий

```bash
git clone https://github.com/KozlovKY/Image-Deblurring.git
cd Image-Deblurring
```

#### 2. Создать окружение через `uv` и установить пакет

```bash
# установить uv, если ещё не установлен
pip install uv
uv venv .venv

# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\Activate.ps1

uv pip install -e .
```

Окружение рассчитано на Python 3.10, а полный список базовых зависимостей зашит в `pyproject.toml`.

---

### Подготовка данных

---

Для работы с датасетом **RSBlur** в репозитории есть скрипты предварительной обработки:

- **`data/preprocessing/extract_datasets.sh`** — распаковка частями заархивированного датасета в `data/datasets/RSBlur/...`;
- **`data/preprocessing/create_npy_files.py`** — конвертация изображений RSBlur в `.npy` для обучения.

Детальная инструкция по шагам приведена в `data/preprocessing/README.md`.

Дополнительно подготовка данных интегрирована в **dvc-пайплайн** (`dvc.yaml`):

- стадия `rsblur_extract` — распаковка шардов из `data/zip` в `data/datasets/RSBlur`;
- стадия `rsblur_npy` — генерация `.npy` файлов.

При запуске обучения `train.py` автоматически вызовет `dvc repro rsblur_npy` (если `dvc` установлен и `dvc.yaml` присутствует), что соответствует требованиям задания по Data management.

---

### Обучение моделей

---

Обучение управляется конфигами **Hydra** (по умолчанию `configs/evssm.yaml`) и `PyTorch Lightning Trainer`.

- **Базовый запуск (EVSSM)**:

```bash
python train.py
```

Hydra создаст рабочую директорию в `./outputs/...`, а гиперпараметры возьмёт из `configs/evssm.yaml`.

- **Примеры оверрайдов из CLI**:

```bash
# изменить число эпох и батч-сайз
python train.py train.epochs=50 data.batch_size=8

# изменить директорию с данными (внутри должны быть подпапки blur/ и sharp/)
python train.py data.data_dir="path/to/data_root"

# запустить альтернативную модель (convnet вместо EVSSM)
python train.py --config-name convnet
```

Lightning‑колбэки для чекпоинтов и мониторинга lr конфигурируются в `configs/evssm.yaml` / `configs/convnet.yaml` (секция `callbacks`).

---
