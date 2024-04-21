import os
import datetime
import shutil

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule

from models.loss import PointnetClsLoss
from utils import sample
from utils.build import build_model, build_dataset
from utils.torch_utils import build_optimizer, build_scheduler


class ClassifierEngine(LightningModule):
    # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.model = build_model(cfg)

    def forward(self, x):
        x = self.model(x)
        return x

    def save_txt_model_info(self):
        with open(os.path.join(self.save_model_dir, 'model.txt'), 'w') as f:
            try:
                from models.segmentation.torch_summary import summary
                model_info = summary(self.model, (6 if self.hparams.use_normals else 3, self.hparams.num_points),
                                     device='cuda' if 'cuda' in str(self.device) else 'cpu')
                f.writelines(model_info)
            except:
                pass
            f.write('\n' + 'Model Structure:\n')
            f.write(str(self.model) + '\n')
        f.close()

    def save_config_file(self):
        shutil.copy(self.hparams.default_config, self.save_config_dir)
        shutil.copy(self.hparams.model_config, self.save_config_dir)
        shutil.copy(self.hparams.dataset_config, self.save_config_dir)

    def add_dict_prefix(self, input_dict: dict, prefix: str):
        new_dict = {}
        for k, v in input_dict.items():
            new_dict[prefix+'_'+k] = v
        return new_dict

    def setup(self, stage=None):
        self.train_dataset = build_dataset(self.hparams, split='train')
        self.val_dataset = build_dataset(self.hparams, split='test')

    def train_dataloader(self):
        self.training_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=True if torch.cuda.is_available() and self.hparams.pin_memory else False
        )
        return self.training_dataloader

    def val_dataloader(self):
        self.validation_dataloader = DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            batch_size=1,
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=True if torch.cuda.is_available() and self.hparams.pin_memory else False
        )
        return self.validation_dataloader

    def training_step(self, batch, batch_idx):
        points, target = batch
        data_type, device = points.dtype, points.device
        points = points.cpu().numpy()
        points = sample.random_point_dropout(points)
        points[:, :, 0:3] = sample.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = sample.shift_point_cloud(points[:, :, 0:3])
        points = torch.from_numpy(points).to(device=device, dtype=data_type)
        points = points.transpose(2, 1)
        pred, trans_feat = self.forward(points)
        loss_inputs = {name: [pred, target, trans_feat] for name in self.hparams.losses}
        loss_result = self.compute_loss(loss_inputs)

        metric_inputs = {metric: [pred, target] for metric in self.hparams.metrics}
        metric_result = self.compute_metrics(metric_inputs)
        log_msg = loss_result  # log losses
        log_msg.update(metric_result)
        log_msg = self.add_dict_prefix(log_msg, 'train')

        for k, v in log_msg.items():
            self.log(k, v, prog_bar=True)
        self.train_step_outputs.append(log_msg)

        if batch_idx % self.hparams.log_txt_interval == 0 or batch_idx == len(self.training_dataloader) - 1:
            with open(os.path.join(self.hparams.work_dir, 'train_log_steps.txt'), 'a') as f:
                str_global_steps = f'global_steps: {self.global_step}'
                str_msg = f'{str_global_steps}\tsteps: [{batch_idx}/{len(self.training_dataloader)-1}]\t'
                for k, v in self.compute_list_dict_avg(self.train_step_outputs).items():  # log_msg.items()
                    str_msg += f'{k}: {v}\t'
                f.write(str_msg + '\n')
            f.close()

        return loss_result['loss']
        # return {'loss': loss, 'train_acc': acc}  # must have key loss

    def validation_step(self, batch, batch_idx):
        points, target = batch
        points = points.transpose(2, 1)
        pred, trans_feat = self.forward(points)
        loss_inputs = {name: [pred, target, trans_feat] for name in self.hparams.losses}
        loss_result = self.compute_loss(loss_inputs)

        metric_inputs = {metric: [pred, target] for metric in self.hparams.metrics}
        metric_result = self.compute_metrics(metric_inputs)
        log_msg = loss_result  # log loss
        log_msg.update(metric_result)
        log_msg = self.add_dict_prefix(log_msg, 'val')

        for k, v in log_msg.items():
            self.log(k, v, prog_bar=True)
        self.validation_step_outputs.append(log_msg)

        if batch_idx % self.hparams.log_txt_interval == 0 or batch_idx == len(self.validation_dataloader) - 1:
            with open(os.path.join(self.hparams.work_dir, 'validation_log_steps.txt'), 'a') as f:
                str_global_steps = f'global_steps: {self.global_step}'
                str_msg = f'{str_global_steps}\tsteps: [{batch_idx}/{len(self.validation_dataloader)-1}]\t'
                for k, v in self.compute_list_dict_avg(self.validation_step_outputs).items():  # log_msg
                    str_msg += f'{k}: {v}\t'
                f.write(str_msg + '\n')
            f.close()

        # return {'val_fitness': fitness}

    def accuracy(self, logits, lables):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), lables).to(torch.float32)) / len(lables)
        return acc

    def plot_keyword_curve(self, txt_file, keyword='loss'):
        with open(txt_file, 'r') as f:
            line_info = [line.strip() for line in f.readlines()]
        f.close()
        steps = list(range(len(line_info)))
        info_dict = {}
        for line in line_info:
            line_dict = {data.split(':')[0]: data.split(':')[1] for data in line.split('\t')}
            for k, v in line_dict.items():
                if keyword in k:
                    if k not in info_dict:
                        info_dict[k] = []
                    info_dict[k].append(float(v))
        # plot
        if len(info_dict) != 0:  # 查询keyword是否存在
            for k, v in info_dict.items():
                # plt.figure(figsize=(12, 8))
                plt.plot(steps, v, label=f'{k}_curve')
                plt.xlabel('steps')  # fontsize change font size
                plt.ylabel(k)
                # plt.title(f'{k}_curve')
                plt.legend()
                plt.savefig(txt_file.split('.txt')[0] + f'_{k}.png')
                # plt.show()

    @property
    def metric_map(self):
        map = {
            'accuracy': self.accuracy
        }
        return map

    @property
    def loss_map(self):
        map = {
            'pointnet_cls': PointnetClsLoss()
        }
        return map

    @property
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def compute_metrics(self, inputs: dict):
        if not isinstance(self.hparams.metrics, (tuple, list)):
            self.hparams.metrics = [self.hparams.metrics]
        if self.hparams.metric_weights is None:
            self.hparams.metric_weights = [1.0 / len(self.hparams.metrics)] * len(self.hparams.metrics)
        else:
            if not isinstance(self.hparams.metric_weights, (tuple, list)):
                self.hparams.metric_weights = [self.hparams.metirc_weights]
            assert sum(self.hparams.metric_weights) == 1.0, 'sum of weights must be set to 1.0.'
        assert len(self.hparams.metrics) == len(self.hparams.metric_weights) == len(inputs)
        result, fitness = {}, 0.0
        for metric, metric_weight in zip(self.hparams.metrics, self.hparams.metric_weights):
            mid = self.metric_map[metric](*inputs[metric])
            result[metric] = mid
            fitness += metric_weight * mid
        result['fitness'] = fitness
        return result

    def compute_loss(self, inputs: dict):
        if not isinstance(self.hparams.losses, (tuple, list)):
            self.hparams.losses = [self.hparams.losses]
        if self.hparams.loss_weights is None:
            self.hparams.loss_weights = [1.0 / len(self.hparams.losses)] * len(self.hparams.losses)
        else:
            if not isinstance(self.hparams.loss_weights, (tuple, list)):
                self.hparams.loss_weights = [self.hparams.loss_weights]
        assert len(self.hparams.losses) == len(self.hparams.loss_weights) == len(inputs)
        result, loss = {}, 0.0
        for name, loss_weight in zip(self.hparams.losses, self.hparams.loss_weights):
            mid = self.loss_map[name](*inputs[name])
            result[name] = mid * loss_weight
            loss += loss_weight * mid
        result['loss'] = loss
        return result

    def record_log_datetime(self):
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        with open(os.path.join(self.hparams.work_dir, 'train_log_steps.txt'), 'a') as f:
            f.write(f'-----------------------------------{time}-----------------------------------\n')
        f.close()

        with open(os.path.join(self.hparams.work_dir, 'train_log_epochs.txt'), 'a') as f:
            f.write(f'-----------------------------------{time}-----------------------------------\n')
        f.close()

        with open(os.path.join(self.hparams.work_dir, 'validation_log_steps.txt'), 'a') as f:
            f.write(f'-----------------------------------{time}-----------------------------------\n')
        f.close()

        with open(os.path.join(self.hparams.work_dir, 'validation_log_epochs.txt'), 'a') as f:
            f.write(f'-----------------------------------{time}-----------------------------------\n')
        f.close()

    def on_train_start(self):
        # general name / experiment name / models
        # save model file, config file, dataset file
        # 保存日志到txt或log或json文件中，记录模型信息，过程信息
        self.save_model_dir = os.path.join(self.hparams.work_dir, 'models')
        self.save_config_dir = os.path.join(self.hparams.work_dir, 'configs')
        os.makedirs(self.save_model_dir, exist_ok=True)
        os.makedirs(self.save_config_dir, exist_ok=True)

        self.save_txt_model_info()
        self.save_config_file()
        if not self.hparams.create_time_dir:
            self.record_log_datetime()

    def on_train_end(self):
        self.plot_keyword_curve(os.path.join(self.hparams.work_dir, 'train_log_steps.txt'), keyword='loss')
        self.plot_keyword_curve(os.path.join(self.hparams.work_dir, 'train_log_epochs.txt'), keyword='loss')
        self.plot_keyword_curve(os.path.join(self.hparams.work_dir, 'train_log_epochs.txt'), keyword='lr')
        if not isinstance(self.hparams.metrics, (tuple, list)):
            self.hparams.metrics = [self.hparams.metrics]
        for metric in self.hparams.metrics:
            self.plot_keyword_curve(os.path.join(self.hparams.work_dir, 'train_log_steps.txt'), keyword=metric)
            self.plot_keyword_curve(os.path.join(self.hparams.work_dir, 'train_log_epochs.txt'), keyword=metric)

    def on_validation_end(self):
        self.plot_keyword_curve(os.path.join(self.hparams.work_dir, 'validation_log_steps.txt'), keyword='loss')
        self.plot_keyword_curve(os.path.join(self.hparams.work_dir, 'validation_log_epochs.txt'), keyword='loss')
        if not isinstance(self.hparams.metrics, (tuple, list)):
            self.hparams.metrics = [self.hparams.metrics]
        for metric in self.hparams.metrics:
            self.plot_keyword_curve(os.path.join(self.hparams.work_dir, 'validation_log_steps.txt'), keyword=metric)
            self.plot_keyword_curve(os.path.join(self.hparams.work_dir, 'validation_log_epochs.txt'), keyword=metric)

    def compute_list_dict_avg(self, list_dict):
        avg_dict = {}
        for key in list_dict[0].keys():
            avg_dict[key] = 0.0
        for data_dict in list_dict:
            for key, value in data_dict.items():
                avg_dict[key] += value
        avg_dict = {k+'_avg': v / len(list_dict) for k, v in avg_dict.items()}
        return avg_dict

    def on_train_epoch_end(self):

        avg_dict = self.compute_list_dict_avg(self.train_step_outputs)
        avg_dict.update({'lr': self.get_lr})  # update learning rate, only training stage
        for k, v in avg_dict.items():
            self.log(k, v, prog_bar=True)
        with open(os.path.join(self.hparams.work_dir, 'train_log_epochs.txt'), 'a') as f:
            str_msg = f'epochs: [{self.current_epoch}/{self.hparams.max_epochs-1}]\t' if self.hparams.max_epochs else f'epochs: {self.current_epoch}\t'
            for k, v in avg_dict.items():
                str_msg += f'{k}: {v}\t'
            f.write(str_msg + '\n')
        f.close()
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        avg_dict = self.compute_list_dict_avg(self.validation_step_outputs)
        for k, v in avg_dict.items():
            self.log(k, v, prog_bar=True)
        with open(os.path.join(self.hparams.work_dir, 'validation_log_epochs.txt'), 'a') as f:
            str_msg = f'epochs: [{self.current_epoch}/{self.hparams.max_epochs-1}]\t' if self.hparams.max_epochs else f'epochs: {self.current_epoch}\t'
            for k, v in avg_dict.items():
                str_msg += f'{k}: {v}\t'
            f.write(str_msg + '\n')
        f.close()
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        self.optimizer = build_optimizer(self)
        scheduler = build_scheduler(self)
        return [self.optimizer], [scheduler]
