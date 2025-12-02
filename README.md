## Image-Deblurring

### Описание

---

Проект для восстановления резкости изображений.

Основные компоненты:

- **PyTorch Lightning** — обучение моделей;
- **Hydra** — управление конфигами и экспериментами;
- **MLflow** — логирование экспериментов;
- **DVC** — версионирование данных;

---

### Структура проекта

---

- **`src/`** — основной код:
  - **`data/`** — датасет, аугментации, DVC;
  - **`models/`** — модели деблюринга (EVSSM, DeblurCNN);
  - **`training/`** — Lightning‑модуль (`LitDeblurring`) и DataModule;
  - **`utils/`** — метрики (PSNR, SSIM), лоссы (PSNRLoss), утилиты;
- **`configs/`** — конфиги **Hydra**;
- **`data/scripts/`** — скрипты подготовки данных:
  - `yandex_download.py` — скачивание архива с Я.Диска;
  - `extract_shards.sh` — распаковка архивов;
  - `split_train_val.py` — разделение на train/val;
- **`dvc.yaml`** — DVC-пайплайн подготовки данных;
- **`train.py`** — входная точка обучения;
- **`pyproject.toml`** — зависимости проекта (uv);
- **`.pre-commit-config.yaml`** — конфигурация pre-commit хуков.

---

### Setup

---

#### 1. Клонировать репозиторий

```bash
git clone https://github.com/KozlovKY/Image-Deblurring.git
cd Image-Deblurring
```

#### 2. Создать окружение через `uv` и установить пакет

Окружение рассчитано на Python 3.11, а полный список базовых зависимостей зашит в `pyproject.toml`.

```bash
pip install uv
uv venv deblur --python 3.11

# Linux / macOS:
source deblur/bin/activate
# Windows:
deblur/bin/activate

uv pip install -e ".[dev]"
```

#### 3. Установить pre-commit хуки

```bash
pre-commit install

pre-commit run -a
```

Используются следующие инструменты для контроля качества кода:

- **ruff** — линтер и форматтер (заменяет black, isort, flake8)
- **pre-commit hooks** — базовые проверки (YAML, trailing whitespace, end-of-file)
- **prettier** — форматирование не-Python файлов

---

### Подготовка данных

---

В данном сетапе используется subset из `RSBlur` датасета, оригинальный датасет весит порядка 100 гб, поэтому для проверки системы он был сокращен.

Подготовка данных интегрирована в **DVC-пайплайн** (`dvc.yaml`) и выполняется автоматически при запуске обучения. Пайплайн состоит из трёх стадий:

1. **`rsblur_download`** — скачивание архива датасета с Я.Диска в `data/zip/`
2. **`rsblur_extract`** — распаковка архивов из `data/zip` в `data/datasets/RSBlur/`
3. **`rsblur_split`** — разделение данных на train/val и создание финального датасета в `data/datasets/rsblur_train_val/`

При запуске `python train.py` автоматически вызывается `dvc repro`, который выполняет все необходимые стадии пайплайна.

**Ручной запуск пайплайна:**

```bash
# Запустить весь пайплайн
dvc repro

# Или запустить конкретную стадию
dvc repro rsblur_download    # только скачивание
dvc repro rsblur_extract      # только распаковка
dvc repro rsblur_split        # только разделение
```

### MLflow Setup

Перед запуском тренировки нужно запустить MLflow сервер:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Сервер будет доступен по адресу `http://127.0.0.1:8080`.

---

### Train

---

#### Обучение моделей

Обучение управляется конфигами **Hydra** (по умолчанию `configs/convnet.yaml`) и `PyTorch Lightning Trainer`.

**Базовый запуск:**

```bash
python train.py
```

При запуске автоматически:

1. Выполняется DVC-пайплайн подготовки данных (скачивание → распаковка → разделение)
2. Создаётся рабочая директория в `outputs/YYYY-MM-DD/HH-MM-SS/`
3. Сохраняются чекпоинты в `outputs/.../checkpoints/`
4. Логируются метрики в MLflow

Возможно установить свои параметры, переписав значения конфигов.
**Примеры оверрайдов из CLI:**

```bash
python train.py train.epochs=50 data.batch_size=8

python train.py --config-name convnet

python train.py model.optimizer.lr=1e-3
```

Можно запустить весь процесс одним скриптом:

```bash
chmod u+x train_e2e.sh
./train_e2e.sh
```

**Примечание:** Если автоматическое скачивание с Яндекс.Диска не работает, скачайте архив вручную по [ссылке](https://disk.360.yandex.ru/d/6LNWs_woE4JWeA) и поместите его в папку `data/zip/`.

**Доступные конфигурации:**

- `configs/evssm.yaml` — модель EVSSM (1.4 M)
- `configs/convnet.yaml` — простая CNN модель (20 K)

Lightning‑колбэки для чекпоинтов и мониторинга lr конфигурируются в соответствующих YAML файлах (секция `callbacks`).

---
