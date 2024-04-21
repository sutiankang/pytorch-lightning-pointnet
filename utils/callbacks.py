import os
import pytorch_lightning.callbacks as plc
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


def load_callbacks(cfg):
    callbacks = []
    if cfg.use_early_stopping:
        callbacks.append(plc.EarlyStopping(
            monitor=cfg.monitor,
            mode=cfg.monitor_mode,
            patience=cfg.patience,
            min_delta=cfg.min_delta
        ))

    callbacks.append(plc.ModelCheckpoint(
        dirpath=os.path.join(cfg.work_dir, 'checkpoints'),
        monitor=cfg.monitor,
        filename='best-{epoch:02d}-{val_fitness_avg:.3f}',
        save_top_k=cfg.save_top_k,
        mode=cfg.monitor_mode,  # select max or min indices
        save_last=cfg.save_last
    ))

    if cfg.scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))

    # pbar = plc.TQDMProgressBar(refresh_rate=1)
    pbar = plc.RichProgressBar(
        refresh_rate=1,
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".4f",  # ".3e"
        )
    )
    callbacks.append(pbar)

    return callbacks