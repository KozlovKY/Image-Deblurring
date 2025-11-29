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
- `train.py` — входная точка обучения (Hydra + PyTorch Lightning)
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

Корневая директория с данными настраивается в конфиге `configs/train.yaml`:

```yaml
data:
  data_dir: "data"  # внутри ожидаются подпапки blur/ и sharp/
```

Позже сюда можно встроить `dvc.api` или `download_data()` для автоматической загрузки.

------------------------------------
### Train
------------------------------------

Обучение управляется конфигом **Hydra** `configs/train.yaml` и `PyTorch Lightning Trainer`. Базовый запуск (EVSSM):

```bash
python train.py
```

Hydra создаст рабочую директорию в `./outputs/...`, а гиперпараметры возьмёт из `configs/train.yaml`.

Примеры полезных оверрайдов прямо из CLI:

```bash
# изменить число эпох и батч-сайз
python train.py train.epochs=50 data.batch_size=8

# изменить директорию с данными (внутри должны быть подпапки blur/ и sharp/)
python train.py data.data_dir="path/to/data_root"

# запустить простой UNet вместо EVSSM
python train.py --config-name unet

------------------------------------
