import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

class EMACallback(pl.Callback):
    def __init__(self, decay=0.0):
        self.decay = decay
        self.ema_state_dict = {}
        self.backup_state_dict = {}
        self.global_step = 0

    def on_fit_start(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.ema_state_dict[name] = param.data.clone().detach()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.global_step += 1
        with torch.no_grad():
            effective_decay = min(self.decay, 1 - 1 / self.global_step)
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    ema = self.ema_state_dict[name]
                    ema.mul_(effective_decay).add_(param.data, alpha=1 - effective_decay)

    def _swap_to_ema(self, pl_module):
        self.backup_state_dict = {}
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.backup_state_dict[name] = param.data.clone()
                param.data.copy_(self.ema_state_dict[name])

    def _restore_original(self, pl_module):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup_state_dict[name])
        self.backup_state_dict = {}

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["ema_state_dict"] = self.ema_state_dict

    def on_validation_start(self, trainer, pl_module):
        self._swap_to_ema(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._restore_original(pl_module)

    def on_test_start(self, trainer, pl_module):
        self._swap_to_ema(pl_module)

    def on_test_end(self, trainer, pl_module):
        self._restore_original(pl_module)


class WarmupEarlyStopping(EarlyStopping):
    def __init__(self, warmup_epochs=0, **kwargs):
        super().__init__(**kwargs)
        self.warmup_epochs = warmup_epochs

    def _run_early_stopping_check(self, trainer):
        if trainer.current_epoch < self.warmup_epochs:
            return
        super()._run_early_stopping_check(trainer)

def add_noise(x, std, p=0.9):
    if std <= 0:
        return x
    mask = (torch.rand_like(x) < p).float()
    noise = torch.randn_like(x) * std
    return x + mask * noise

def gene_add_noise(x, std, p=0.9):
    if std <= 0:
        return x

    nonzero_mask = (x != 0)
    prob_mask = (torch.rand_like(x) < p)

    mask = nonzero_mask & prob_mask
    noise = torch.randn_like(x) * std

    return x + noise * mask


def feature_dropout(x, p):
    if p <= 0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask

def gene_feature_dropout(x, p):
    if p <= 0:
        return x

    nonzero_mask = (x != 0)
    drop_mask = (torch.rand_like(x) > p)
    final_mask = nonzero_mask & drop_mask

    return x * final_mask


def random_neighbor_sampling(indices, fixed_num, random_num):
    fixed = indices[:fixed_num]
    remain = indices[fixed_num:]

    
    if random_num > 0 and remain.numel() > 0:
        perm = torch.randperm(remain.size(0))[:random_num]
        sampled = remain[perm]
    else:
        sampled = torch.tensor([], dtype=indices.dtype, device=indices.device)

    
    combined = torch.cat([fixed, sampled], dim=0)

    
    if combined.size(0) > 1:
        first = combined[0:1]
        rest = combined[1:]
        perm_rest = torch.randperm(rest.size(0))
        rest_shuffled = rest[perm_rest]
        combined = torch.cat([first, rest_shuffled], dim=0)

    return combined


def random_patch_sampling(samples, keep_num=64):
    N = samples.shape[0]

    perm = torch.randperm(N)

    if keep_num < N:
        perm = perm[:keep_num]

    sampled = samples[perm]
    return sampled

