import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig

from src.data.download import ensure_data


@hydra.main(version_base=None, config_path="configs", config_name="evssm")
def main(cfg: DictConfig) -> None:
    """Entry point for training with Lightning + Hydra.

    Checkpointing and logging follow the recommended Lightning patterns:
    - automatic best/last checkpoints via ModelCheckpoint callback;
    - metrics logged through Lightning to the configured logger (MLflow).
    """
    # reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Optimize for Tensor Cores on CUDA devices
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    ensure_data(stage=None)

    datamodule = hydra.utils.instantiate(cfg.data_module)

    model = hydra.utils.instantiate(cfg.model)

    # mlflow logger (if enabled in config)
    logger = hydra.utils.instantiate(cfg.mlflow_logger)
    # instantiate callbacks from config if present
    callbacks = [
        hydra.utils.instantiate(cb_cfg)
        for cb_cfg in (getattr(cfg, "callbacks", {}) or {}).values()
    ]

    # trainer
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
