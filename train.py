import argparse
import torch

from utils.utils import read_file, create_work_dir, merge_args_with_dict
from utils.callbacks import load_callbacks
from utils.build import build_model_engine

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--default_config', type=str, help='',
                        default='configs/default.yaml')
    parser.add_argument('--dataset_config', type=str, help='',
                        default='configs/datasets/modelnet40_normal_resampled.yaml')
    parser.add_argument('--model_config', type=str, help='',
                        default='configs/models/pointnet_cls.yaml')
    cfg = parser.parse_args()

    default_config_dict = read_file(cfg.default_config)
    dataset_config_dict = read_file(cfg.dataset_config)
    model_config_dict = read_file(cfg.model_config)

    cfg = merge_args_with_dict(cfg, [default_config_dict, dataset_config_dict, model_config_dict])
    cfg = create_work_dir(cfg)

    return cfg


if __name__ == '__main__':
    cfg = get_cfg()

    pl.seed_everything(seed=cfg.seed, workers=cfg.workers)
    callbacks = load_callbacks(cfg)

    model = build_model_engine(cfg)

    logger1 = TensorBoardLogger(
        save_dir=cfg.work_dir,
        name='tensorboard',
        default_hp_metric=False
    )
    logger2 = CSVLogger(save_dir=cfg.work_dir, name='csv')

    trainer = Trainer(
        min_epochs=cfg.min_epochs,
        max_epochs=cfg.max_epochs,
        callbacks=callbacks,
        precision=cfg.precision,
        default_root_dir=cfg.work_dir,
        logger=[logger1, logger2],
        enable_model_summary=cfg.model_summary,
        accelerator=cfg.accelerator,
        devices=cfg.num_gpus,
        num_sanity_val_steps=1,
        benchmark=True if cfg.benchmark and torch.cuda.is_available() else False,
        deterministic=cfg.deterministic,
        profiler=cfg.profiler,
        strategy=cfg.strategy,
        sync_batchnorm=cfg.sync_bn,
        check_val_every_n_epoch=cfg.eval_interval,
        gradient_clip_val=cfg.gradient_clip,
        gradient_clip_algorithm=cfg.gradient_clip_algorithm,
        accumulate_grad_batches=cfg.accumulate_gradient
    )
    # if load checkpoint
    if cfg.checkpoint:
        model.load_from_checkpoint(checkpoint_path=cfg.checkpoint,
                                   map_location=cfg.map_location,
                                   strict=cfg.strict)
    trainer.fit(model, ckpt_path=cfg.resume)  # resume from checkpoint
