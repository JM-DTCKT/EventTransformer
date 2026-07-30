import os
import time

from torch import nn
import torch.nn.functional as F
import torch

import pytorch_lightning as pl

from pytorch_lightning import Trainer, LightningModule
from torch.optim import lr_scheduler

from data_generation import Event_DataModule
import evaluation_utils

from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import training_utils
import json
import pandas as pd
import numpy as np
import copy
from torch.optim import AdamW

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.EvT import CLFBlock, MLPBlock
from models.EvT import EvNetBackbone


class TrainingMonitorCallback(Callback):
    """에포크별 요약 출력 + 학습 곡선 이미지 저장"""

    def __init__(self, save_dir, total_epochs):
        self.save_dir = save_dir
        self.total_epochs = total_epochs
        self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}
        self._train_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if self._train_start_time is None:
            self._train_start_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        self.history['epoch'].append(epoch)
        for src, dst in [('train_loss_total', 'train_loss'),
                         ('val_loss_total',   'val_loss'),
                         ('val_acc',          'val_acc')]:
            if src in metrics:
                self.history[dst].append(float(metrics[src]))

        done    = epoch + 1
        remain  = self.total_epochs - done
        bar_len = 40
        filled  = int(bar_len * done / self.total_epochs)
        bar     = '█' * filled + '░' * (bar_len - filled)

        elapsed_total = time.time() - self._train_start_time if self._train_start_time else 0
        avg_ep_sec    = elapsed_total / done if done > 0 else 0
        eta_sec       = avg_ep_sec * remain

        def _fmt(sec):
            h, m = divmod(int(sec), 3600)
            m, s = divmod(m, 60)
            return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

        eta_str     = _fmt(eta_sec)
        elapsed_str = _fmt(elapsed_total)

        tr   = f"tr_loss={self.history['train_loss'][-1]:.4f}" if self.history['train_loss'] else ''
        val  = f"val_loss={self.history['val_loss'][-1]:.4f}"  if self.history['val_loss']   else ''
        acc  = f"val_acc={self.history['val_acc'][-1]:.4f}"    if self.history['val_acc']    else ''
        best = f"best={max(self.history['val_acc']):.4f}"       if self.history['val_acc']    else ''
        print(f"\n[{bar}] {done:>3}/{self.total_epochs}  elapsed={elapsed_str}  ETA={eta_str}  {tr}  {val}  {acc}  {best}",
              flush=True)

        self._save_plot()

    def _save_plot(self):
        if not self.history['epoch']:
            return
        epochs = self.history['epoch']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if self.history['train_loss']:
            axes[0].plot(epochs[:len(self.history['train_loss'])],
                         self.history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
        if self.history['val_loss']:
            axes[0].plot(epochs[:len(self.history['val_loss'])],
                         self.history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if self.history['val_acc']:
            axes[1].plot(epochs[:len(self.history['val_acc'])],
                         self.history['val_acc'], 'g-', label='Val Accuracy', linewidth=1.5)
            best_acc = max(self.history['val_acc'])
            best_ep  = epochs[self.history['val_acc'].index(best_acc)]
            axes[1].axhline(y=best_acc, color='g', linestyle='--', alpha=0.5,
                            label=f'Best {best_acc:.4f} (ep {best_ep})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            f'Training Progress  |  Epoch {epochs[-1] + 1} / {self.total_epochs}',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_progress.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)



class EvNetModel(LightningModule):

    def __init__(self, backbone_params, clf_params, optim_params, loss_weights=None):
        super().__init__()
        self.save_hyperparameters()

        self.backbone_params = backbone_params
        self.clf_params = clf_params
        self.optim_params = optim_params
        
        # Initialize Backbone
        self.backbone = EvNetBackbone(**backbone_params)
        # Initialize classifier
        self.clf_params['ipt_dim'] = self.backbone_params['embed_dim']
        # TODO: move to single variable
        self.models_clf = nn.ModuleDict([ [str(0),CLFBlock(**self.clf_params)] ])
        # self.models_clf = CLFBlock(**self.clf_params)
        
        self.loss_weights = loss_weights
        self.init_optimizers()
        
       
    def init_optimizers(self):
        from torchmetrics import Accuracy
        self.criterion = nn.NLLLoss(weight = self.loss_weights)
        self.accuracy = Accuracy(task='multiclass', num_classes=self.clf_params['opt_classes'])
        

    def forward(self, x, pixels):
        # Get updated latent vectors
        embs = self.backbone(x, pixels)
        # Get latent vectors classification
        clf_logits = torch.stack([ self.models_clf[str(0)](embs) ]).mean(axis=0)
        return embs, clf_logits
    
        
    def configure_optimizers(self):

        # Import base optimizer
        base_optim = AdamW
        optim = base_optim(self.parameters(), **self.optim_params['optim_params'])
    
        if 'scheduler' in self.optim_params: 
            if self.optim_params['scheduler']['name'] == 'lr_on_plateau': 
                sched = lr_scheduler.ReduceLROnPlateau(optim, **self.optim_params['scheduler']['params'])
            elif self.optim_params['scheduler']['name'] == 'one_cycle_lr': 
                sched = lr_scheduler.OneCycleLR(optim, max_lr=self.optim_params['optim_params']['lr'],  **self.optim_params['scheduler']['params'])
            return {'optimizer': optim, 'lr_scheduler': sched, 'monitor': self.optim_params['monitor']}
        return optim
    

    # Forward data and calculate loss and acc
    def step(self, polarity, pixels, y):
        embs, clf_logits = self(polarity, pixels)

        loss_clf, loss_contr = 0.0, 0.0
        logs = {}
            
        loss_clf = self.criterion(clf_logits, y)
        preds = torch.argmax(clf_logits, dim=-1)
        
        acc = self.accuracy(preds, y)
 
        logs['loss_clf'] = loss_clf
        logs['acc'] = acc
        
     
        logs['loss_total'] = loss_clf + loss_contr
        
        return logs
        

    def training_step(self, batch, batch_idx):
        #  batch_data -> (#imesteps, batch_size, #events, 2) - (#imesteps, batch_size, #events, 2) - (batch_size)
        polarity, pixels, y = batch    
        losses = self.step(polarity, pixels, y)
        for k,v in losses.items():
            self.log(f'train_{k}', v, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return losses['loss_total']


    def validation_step(self, batch, batch_idx):
        polarity, pixels, y = batch
        losses = self.step(polarity, pixels, y)
        for k,v in losses.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        
        return losses['loss_total']




def train(folder_name, path_results, data_params, backbone_params, clf_params, 
          training_params, optim_params, callback_params, logger_params):

    torch.set_float32_matmul_precision('high')

    # Create the folder where to store the training results
    path_model = training_utils.create_model_folder(path_results, folder_name)

    # PL 1.9.x compatibility
    training_params = training_params.copy()
    if 'gpus' in training_params:
        gpus = training_params.pop('gpus')
        training_params['accelerator'] = 'gpu'
        training_params['devices'] = [int(g) for g in str(gpus).split(',') if g.strip()]

    callbacks = []
    for k, params in callback_params:
        if k == 'early_stopping':
            pass  # early stopping disabled - train full epochs
        if k == 'lr_monitor': callbacks.append(LearningRateMonitor(**params))
        if k == 'model_chck':
            params['dirpath'] = params['dirpath'].format(path_model)
            if 'period' in params:
                params['every_n_epochs'] = params.pop('period')
            params['verbose'] = False
            callbacks.append(ModelCheckpoint(**params))
        
    loggers = []
    if 'csv' in logger_params: 
        logger_params['csv']['save_dir'] = logger_params['csv']['save_dir'].format(path_model)
        loggers.append(CSVLogger(**logger_params['csv']))

    
    # =============================================================================
    # Train
    # =============================================================================
    dm = Event_DataModule(**data_params)
    backbone_params['token_dim'] = dm.token_dim
    clf_params['opt_classes'] = dm.num_classes
    
    if 'pos_encoding' in backbone_params and backbone_params['pos_encoding']['params'].get('shape', -1) == -1:
        backbone_params['pos_encoding']['params']['shape'] = (dm.width, dm.height)
    if backbone_params['downsample_pos_enc'] == -1: backbone_params['downsample_pos_enc'] = data_params['patch_size']
    
    if optim_params['scheduler']['name'] == 'one_cycle_lr':
        optim_params['scheduler']['params']['steps_per_epoch'] = 1
        sched_epochs = optim_params['scheduler']['params'].get('epochs')
        if sched_epochs:
            print(f" - Setting max_epochs to [{sched_epochs}], according to [one_cycle_lr]")
            training_params['max_epochs'] = sched_epochs

    total_epochs = training_params.get('max_epochs', 500)
    callbacks.insert(0, TrainingMonitorCallback(path_model, total_epochs))

    model = EvNetModel(backbone_params=copy.deepcopy(backbone_params), 
                       clf_params=copy.deepcopy(clf_params), 
                       optim_params=copy.deepcopy(optim_params),
                       loss_weights = None if not data_params['balance'] else dm.train_dataloader().dataset.get_class_weights()
                       )
    
    if training_params.pop('stochastic_weight_avg', False):
        from pytorch_lightning.callbacks import StochasticWeightAveraging

        swa_epoch_start = 0.8
        base_lr = optim_params['optim_params']['lr']

        if optim_params['scheduler']['name'] == 'one_cycle_lr':
            # one_cycle_lr는 학습 후반부에 lr을 아주 작은 값까지 감쇠시킨다.
            # swa_lrs를 그와 무관한 큰 고정값(예: 1e-2)으로 주면 SWA가 시작되는
            # 시점에 lr이 스케줄러가 감쇠시켜 놓은 값보다 훨씬 커서 갑자기
            # 튀어 올라 loss가 발산해버린다 (실제로 epoch 192 부근에서 관찰됨).
            # 따라서 SWA 시작 시점에 one_cycle_lr이 도달했을 lr을 미리 계산해
            # 그대로 swa_lrs로 사용해 충돌 없이 이어지도록 한다.
            dummy_optim = AdamW([nn.Parameter(torch.zeros(1))], lr=base_lr)
            dummy_sched = lr_scheduler.OneCycleLR(dummy_optim, max_lr=base_lr, **optim_params['scheduler']['params'])
            swa_start_step = max(1, int(total_epochs * swa_epoch_start))
            for _ in range(swa_start_step):
                dummy_sched.step()
            swa_lrs = dummy_sched.get_last_lr()[0]
        else:
            swa_lrs = base_lr * 0.1

        print(f" - SWA enabled: swa_lrs={swa_lrs:.3e}, swa_epoch_start={swa_epoch_start} (one_cycle_lr 감쇠값에 맞춤)")
        callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lrs, swa_epoch_start=swa_epoch_start))
    trainer = Trainer(**training_params, callbacks=callbacks, logger=loggers)
    
    # Save all params
    json.dump({'data_params': data_params, 'backbone_params': backbone_params, 'clf_params': clf_params, 
               'training_params': training_params,
               'optim_params': optim_params, 'callbacks_params': callback_params, 'logger_params': logger_params},
              open(path_model+'all_params.json', 'w'))
    
    
    trainer.fit(model, dm)
    
    print(' ** Train finished:', path_model)

    logs = evaluation_utils.load_csv_logs_as_df(path_model)
    val_acc = logs[~logs['val_acc'].isna()]['val_acc']
    print(' - Max. Accuracy: {:.4f}'.format(val_acc.values.max()))
    
    for c in [ c for c in logs.columns if 'val_' in c and 'acc' not in c ]:
        v = logs[~logs[c].isna()][c]
        v = v.values.min() if len(v) > 0 else 0.0
        print(' - Min. [{}]: {:.4f}'.format(c, v))
    print("path_model = '{}'".format(path_model))
    
    return path_model

    
