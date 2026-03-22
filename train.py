import os
import pytorch_lightning as pl
from dataset.dataset import MineDataset, load_data
from models.cls_model import Classifier
from models.ctt_model import ContrastiveModel
from torch.utils.data import DataLoader

class TrainingManager:
    def __init__(self, train_names, val_names, cfg, ema_cb, seed=None):
        self.train_names = train_names
        self.val_names = val_names
        self.cfg = cfg
        self.seed = seed
        self.ema_cb = ema_cb

        self.ctt_model = ContrastiveModel(self.cfg.ctt, self.cfg.augment)
        self.cls_model = Classifier(self.ctt_model, self.cfg.cls, self.cfg.augment)

        self.val_data = load_data(self.val_names, cfg)
        self.train_data = load_data(self.train_names, cfg)

    def get_dataloader(self, mode, batch_size, is_train):
        if is_train:
            dataset = MineDataset(self.train_data, mode, cfg=self.cfg, is_train=is_train)
            dataloader = DataLoader(
                dataset, batch_size=batch_size,
                shuffle=True, num_workers=self.cfg.dataset.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )
        else:
            dataset = MineDataset(self.val_data, mode, cfg=self.cfg, is_train=False)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return dataloader

    def _train_model(self, model, train_dataloader, epoch):
        print(os.getcwd())
        if self.seed is not None:
            log_dir = f'{os.getcwd()}/{self.seed}/{self.val_names[0]}'
        else :
            log_dir = f'{os.getcwd()}/{self.val_names[0]}'
        os.makedirs(log_dir, exist_ok=True)

        callbacks = []
        callbacks.append(self.ema_cb)

        trainer = pl.Trainer(
            max_epochs=epoch,
            precision=16,
            default_root_dir=log_dir,
            callbacks=callbacks,
        )
        trainer.fit(model, train_dataloaders=train_dataloader)

    def train_ctt(self):
        train_dataloader = self.get_dataloader(
            mode='ctt',
            batch_size=self.cfg.ctt.training.batch_size,
            is_train=True,
        )
        self._train_model(self.ctt_model, train_dataloader, self.cfg.ctt.training.epoch)

    def train_cls(self):
        train_dataloader = self.get_dataloader(
            mode='cls',
            batch_size=self.cfg.cls.training.batch_size,
            is_train=True,
        )
        self._train_model(self.cls_model, train_dataloader, self.cfg.cls.training.epoch)


    def train(self):
        self.train_ctt()
        self.train_cls()
