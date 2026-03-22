import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, AUROC, AveragePrecision
from adan import Adan
from models.module import SelfAttention, AttentionLayer
from utils import add_noise, feature_dropout


class Classifier(pl.LightningModule):
    def __init__(self, ctt_model, cls_cfg, aug):
        super().__init__()
        self.save_hyperparameters()
        self.cls_cfg = cls_cfg
        self.aug = aug
        self.ctt_model = ctt_model
        ctt_dim = int(ctt_model.ctt_dim)
        patch_dim = int(ctt_model.patch_dim)
        num_layers = int(cls_cfg.model.num_layers)
        num_heads = int(cls_cfg.model.num_heads)
        dropout = float(cls_cfg.training.dropout)
        cls_dropout = float(cls_cfg.training.cls_dropout)

        self.selfAttentionEncoder = SelfAttention(
            num_layers=num_layers,
            embed_dim=ctt_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.patch_projection = nn.Sequential(
            nn.Linear(patch_dim, ctt_dim),
        )

        self.cross_attention = AttentionLayer(
            embed_dim=ctt_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.gate1 = nn.Sequential(
            nn.Linear(ctt_dim * 2, ctt_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )

        self.gate2 = nn.Sequential(
            nn.Linear(ctt_dim * 2, ctt_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(ctt_dim * 2, ctt_dim),
            nn.LayerNorm(ctt_dim),
            nn.GELU(),
            nn.Dropout(cls_dropout),

            nn.Linear(ctt_dim, ctt_dim // 4),
            nn.LayerNorm(ctt_dim // 4),
            nn.GELU(),
            nn.Dropout(cls_dropout),

            nn.Linear(ctt_dim // 4, 1),
        )

        self.acc = BinaryAccuracy()
        self.auc = AUROC(task="binary")
        self.ap = AveragePrecision(task="binary")
        self.f1 = BinaryF1Score()

        self.label_score = []
        self.test_outputs = []
        self.result = {}

    def forward(self, batch):
        """
        The forward propagation process

        Parameters:
            batch (dict):{
                cls: (batch_size, nb_num, patch_dim),
                patch_tokens: (batch_size, cls + reg num, patch_dim),
                gene: (batch_size, nb_num, gene_dim),
                label: (batch_size, 1),
            }

        Returns: (predicted score) : shape(batch_size, 1)
        """
        
        aligned_gene, aligned_cls = self._extract_aligned_features(batch)

        
        macro_gene, macro_cls = self._MacroRE(aligned_gene, aligned_cls)

        
        micro_rep = self._MicroRE(batch, aligned_gene)

        
        unified_rep = self._MSGRI(micro_rep, macro_gene, macro_cls)

        
        return self.classifier(unified_rep)

    def _extract_aligned_features(self, batch):
        return self.ctt_model.neighbor_forward(batch)

    def _MacroRE(self, aligned_gene, aligned_cls):
        nb_num = aligned_gene.shape[1]
        output = self.selfAttentionEncoder(torch.cat([aligned_gene, aligned_cls], dim=1))
        macro_gene, macro_cls = output[:, 0, :], output[:, nb_num, :]
        return macro_gene, macro_cls

    def _MicroRE(self, batch, aligned_gene):
        patch_tokens = batch["patch_tokens"]
        batch_size, sub_patch_num, patch_dim = patch_tokens.shape
        patch_proj = self.patch_projection(patch_tokens.reshape(-1, patch_dim))
        patch_proj = patch_proj.reshape(batch_size, sub_patch_num, -1)

        ca_out = self.cross_attention(aligned_gene[:, [0], :], patch_proj, patch_proj)
        return ca_out.squeeze(1)


    def _MSGRI(self, micro_rep, macro_gene, macro_cls):
        
        gate1 = self.gate1(torch.cat([micro_rep, macro_gene], dim=-1))
        gate2 = self.gate2(torch.cat([micro_rep, macro_cls], dim=-1))

        
        fusion1 = gate1 * micro_rep +  (1 - gate1) * macro_gene
        fusion2 = gate2 * micro_rep +  (1 - gate2) * macro_cls

        
        unified_rep = torch.cat([fusion1, fusion2], dim=-1)
        return unified_rep

    def training_step(self, batch, batch_idx):
        """
        training process

        Parameters:
            batch (dict):{
                cls: (batch_size, nb_num, patch_dim),
                patch_tokens: (batch_size, sub_patch_num, patch_dim),
                gene: (batch_size, nb_num, gene_dim),
                label: (batch_size, 1),
            }
            batch_idx (int): Batch index, provided by PyTorch Lightning

        Returns:
            loss
        """
        if self.training:
            batch = self._augment(batch)
        final_cls = self(batch)  
        label = batch["label"].float() 


        epsilon = 0.05  
        label_smooth = label * (1 - epsilon) + 0.5 * epsilon
        total_loss = F.binary_cross_entropy_with_logits(final_cls, label_smooth)

        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True)
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("t_acc_f", self.acc(final_cls, label.int()), prog_bar=True)
        self.log("t_auc_f", self.auc(final_cls, label.int()), prog_bar=True)
        self.log("t_ap_f", self.ap(final_cls, label.int()), prog_bar=True)
        self.log("t_f1_f", self.f1(final_cls, label.int()), prog_bar=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        """
        Test process
        """
        final_cls = self(batch)  
        label = batch["label"].float() 
        final_loss = F.binary_cross_entropy_with_logits(final_cls, label)
        total_loss = final_loss

        self.test_outputs.append({
            "test_loss": total_loss.detach(),
            "final_cls": final_cls.detach(),
            "label": label.detach()
        })


    def on_test_epoch_end(self):
        test_loss = torch.stack([x["test_loss"] for x in self.test_outputs]).mean()
        final_cls = torch.cat([x["final_cls"] for x in self.test_outputs], dim=0)
        label = torch.cat([x["label"] for x in self.test_outputs], dim=0)

        tt_acc_f = self.acc(final_cls, label.int())
        tt_auc_f = self.auc(final_cls, label.int())
        tt_ap_f = self.ap(final_cls, label.int())
        tt_f1_f = self.f1(final_cls, label.int())

        
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("tt_acc_f", self.acc(final_cls, label.int()), prog_bar=True)
        self.log("tt_auc_f", self.auc(final_cls, label.int()), prog_bar=True)
        self.log("tt_ap_f", self.ap(final_cls, label.int()), prog_bar=True)
        self.log("tt_f1_f", self.f1(final_cls, label.int()), prog_bar=True)

        self.label_score = {
            'label': label.int(),
            'score': torch.sigmoid(final_cls),
        }

        self.result = {
            "test_acc_f": float(tt_acc_f),
            "test_auc_f": float(tt_auc_f),
            "test_ap_f": float(tt_ap_f),
            "test_f1_f": float(tt_f1_f),
        }

    def _augment(self, batch):
        return {
            **batch,
            'cls': self._augment_cls(batch['cls']),
            'gene': self._augment_gene(batch['gene']),
            'patch_tokens': self._augment_patch_tokens(batch['patch_tokens'])
        }

    def _augment_patch_tokens(self, x):
        x = add_noise(x, self.aug.patch_noise_std)
        x = feature_dropout(x, self.aug.patch_dropout_p)
        return x

    def _augment_cls(self, x):
        x = add_noise(x, self.aug.cls_noise_std)
        x = feature_dropout(x, self.aug.cls_dropout_p)
        return x

    def _augment_gene(self, x):
        x = add_noise(x, self.aug.gene_noise_std)
        x = feature_dropout(x, self.aug.gene_dropout_p)
        return x

    def configure_optimizers(self):
        return Adan(
            self.parameters(),
            lr=self.cls_cfg.training.lr,
            weight_decay=self.cls_cfg.training.weight_decay,
            max_grad_norm=1.0
        )






