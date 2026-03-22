import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from utils import add_noise, feature_dropout, gene_add_noise, gene_feature_dropout


class ContrastiveModel(pl.LightningModule):
    def __init__(self, ctt_cfg, aug):
        super().__init__()
        self.ctt_cfg = ctt_cfg
        self.aug = aug
        self.ctt_dim = int(ctt_cfg.model.ctt_dim)
        self.patch_dim = int(ctt_cfg.model.patch_dim)
        gene_dim = int(ctt_cfg.model.gene_dim)
        patch_dim = int(ctt_cfg.model.patch_dim)
        dropout = float(ctt_cfg.training.dropout)

        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, self.ctt_dim),
            nn.LayerNorm(self.ctt_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(self.ctt_dim, self.ctt_dim),
            nn.LayerNorm(self.ctt_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(self.ctt_dim, self.ctt_dim),
        )

        self.cls_encoder = nn.Sequential(
            nn.Linear(patch_dim, self.ctt_dim),
            nn.LayerNorm(self.ctt_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(self.ctt_dim, self.ctt_dim),
            nn.LayerNorm(self.ctt_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(self.ctt_dim, self.ctt_dim),
        )

        
        self.loss_weight = {
            'g2p': 0.5,
            'p2g': 0.5
        }

    def neighbor_forward(self, batch):
        
        cls_embedding = batch["cls"]  
        gene = batch["gene"]  
        batch_size, nb_num, gene_dim = gene.shape
        patch_dim = cls_embedding.shape[-1]

        
        
        gene_proj = self.gene_encoder(gene.reshape(-1, gene_dim))  
        gene_proj = gene_proj.reshape(batch_size, nb_num, -1)  

        
        cls_proj = self.cls_encoder(cls_embedding.reshape(-1, patch_dim))  
        cls_proj = cls_proj.reshape(batch_size, nb_num, -1)  

        return gene_proj, cls_proj

    def forward(self, batch):
        
        cls_embedding = batch["cls"]  
        gene = batch["gene"]  

        
        
        gene_proj = self.gene_encoder(gene)  

        
        patch_proj = self.cls_encoder(cls_embedding)  

        return gene_proj, patch_proj


    def training_step(self, batch, batch_idx):
        if self.training:
            batch = self._augment(batch)
        
        gene_proj, patch_proj = self(batch)

        loss = self.contrastive_loss(gene_proj, patch_proj)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def contrastive_loss(self, gene_proj, patch_proj):

        gene_proj = F.normalize(gene_proj, p=2, dim=-1)
        patch_proj = F.normalize(patch_proj, p=2, dim=-1)

        
        similarity = torch.matmul(gene_proj, patch_proj.T)  

        
        similarity /= self.ctt_cfg.training.temperature

        
        labels = torch.arange(gene_proj.shape[0]).to(similarity.device)

        
        g2p_loss = F.cross_entropy(similarity, labels)
        p2g_loss = F.cross_entropy(similarity.T, labels)

        loss = self.loss_weight['g2p'] * g2p_loss + self.loss_weight['p2g'] * p2g_loss
        return loss

    def _augment(self, batch):
        return {
            **batch,
            'cls': self._augment_cls(batch['cls']),
            'gene': self._augment_gene(batch['gene']),
        }

    def _augment_cls(self, x):
        x = add_noise(x, self.aug.cls_noise_std)
        x = feature_dropout(x, self.aug.cls_dropout_p)
        return x

    def _augment_gene(self, x):
        x = gene_add_noise(x, self.aug.gene_noise_std)
        x = gene_feature_dropout(x, self.aug.gene_dropout_p)
        return x

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.ctt_cfg.training.lr, weight_decay=self.ctt_cfg.training.weight_decay)


