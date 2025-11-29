## Blur-removal / Image Deblurring

------------------------------------
### Структура проекта
------------------------------------

- `src/` — python‑пакет с кодом:
  - `data/` — датасеты и аугментации (albumentations)
  - `models/` — модели деблюринга (UNet, Attention UNet, EVSSM и др.)
  - `utils/` — функции обучения, валидации, метрики, лоссы
  - `scripts/` — вспомогательные скрипты для инференса/оценки
- `configs/` — конфиги **Hydra** (гиперпараметры данных, модели, обучения, логирования)
- `train.py` — входная точка обучения (Hydra + mlflow)
- `Task.md` — формулировка задания

------------------------------------
### Setup
------------------------------------

#### 1. Клонировать репозиторий

```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
cd Image-Deblurring
```

#### 2. Создать conda‑окружение из `env.yml`

```bash
conda env create -f env.yml      # или mamba env create -f env.yml
conda activate image-deblurring
```

Окружение настроено под Python 3.10 и включает все основные зависимости проекта
(`torch`, `torchvision`, albumentations, scikit-image, mlflow, wandb и т.д.),
перечисленные в `setup.py` / `pyproject.toml` и используемые в скриптах.

После создания окружения можно сразу запускать обучение.


#### 4. Подготовка данных

Ожидается, что данные лежат в виде пар размытого и резкого изображения (png/jpg):

```text
Image-Deblurring/
  data/
    blur/
      0001.png
      0002.png
      ...
    sharp/
      0001.png
      0002.png
      ...
```

Пути к папкам `data/blur` и `data/sharp` настраиваются в конфиге `configs/train.yaml`:

```yaml
data:
  blur_dir: "data/blur"
  sharp_dir: "data/sharp"
```

Позже сюда можно встроить `dvc.api` или `download_data()` для автоматической загрузки.

------------------------------------
### Train
------------------------------------

Обучение управляется конфигом **Hydra** `configs/train.yaml`. Базовый запуск:

```bash
python train.py
```

Hydra создаст рабочую директорию в `./outputs/...`, а гиперпараметры возьмёт из `configs/train.yaml`.

Примеры полезных оверрайдов прямо из CLI:

```bash
# изменить число эпох и батч-сайз
python train.py train.epochs=50 data.batch_size=8
------------------------------------
### Logging (mlflow)
------------------------------------

Проект логирует метрики и гиперпараметры в **mlflow** (минимум 3 графика: loss, PSNR, SSIM).

Адрес трекинг‑сервера задаётся в `configs/train.yaml`:

```yaml
logging:
  mlflow:
    enable: true
    tracking_uri: "http://127.0.0.1:8080"
    experiment_name: "image-deblurring"
```

Перед запуском обучения нужно поднять mlflow‑сервер (для локальных тестов):

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Далее запуски `python train.py` будут:

- логировать метрики (`train_loss`, `val_loss`, `train_psnr`, `val_psnr`, `train_ssim`, `val_ssim`);
- логировать основные гиперпараметры (`seed`, `batch_size`, `lr`, `epochs`, `model_name`);
- сохранять чекпоинты моделей и складывать их в артефакты mlflow.

------------------------------------
