from pathlib import Path

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # data module
    datamodule = hydra.utils.instantiate(cfg.data_module)

    # lightning model (внутри уже зашита базовая модель, оптимизатор и шедулер)
    model = hydra.utils.instantiate(cfg.model)

    # mlflow logger (если включён в конфиге)
    logger = None
    if cfg.logging.mlflow.enable:
        logger = hydra.utils.instantiate(cfg.mlflow_logger)

    # trainer
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    # запуск обучения
    trainer.fit(model=model, datamodule=datamodule)

    # Для удобства сохраняем финальный чекпоинт в указанную папку
    save_dir = Path(cfg.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(
        str(save_dir / f"{type(model.net).__name__.lower()}_final.ckpt")
    )


if __name__ == "__main__":
    main()
