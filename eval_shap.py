import logging
import os

import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything
import torch

from data.datamodule import USRDataModule
from shap_evaluator import SHAPEvaluator

logging.getLogger("lightning").propagate = False


@hydra.main(config_path="conf", config_name="config_shap")
def main(cfg):
    cfg.gpus = torch.cuda.device_count()

    wandb_logger = None
    if cfg.log_wandb:
        os.environ["WANDB_SILENT"] = "true"
        wandb_logger = instantiate(cfg.logger)

    data_module = USRDataModule(cfg)
    evaluator = SHAPEvaluator(cfg)

    trainer = Trainer(
        **cfg.trainer,
        logger=wandb_logger,
    )

    trainer.test(evaluator, datamodule=data_module)


if __name__ == "__main__":
    main()
