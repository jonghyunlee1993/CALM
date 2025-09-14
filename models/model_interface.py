import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from collections import defaultdict

from models.loss_function import NLLSurvLoss
from utils.eval_util import compute_c_index


class ModelInterface(pl.LightningModule):
    def __init__(
        self, 
        model,
        learning_rate,
        cont_loss="cosine",
        loss_lambda=0.1,
        **kwargs
    ):
        super().__init__()
        self.model = model

        self.surv_loss_fn = NLLSurvLoss()
        self.cont_loss_fn = cont_loss
        if self.cont_loss_fn == "cosine":
            self.contrast_loss_fn = torch.nn.CosineEmbeddingLoss(reduction='none')
        else:
            raise ValueError(f"Unsupported cont_loss: {self.cont_loss_fn}")

        self.loss_lambda = float(loss_lambda)
        self.learning_rate = float(learning_rate)
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.warmup_epochs = kwargs.get('warmup_epochs', 10)

        self.valid_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)

    def forward(self, batch):
        image_feat, text_feat, diagnostic_description, omic_feat, event_time, label, c, source = batch

        logit, mixed_feat, diag_desc_emb, image_CLS, text_CLS, _, _ = self.model(
            image_feat, text_feat, diagnostic_description
        )

        surv_loss = self.surv_loss_fn(logit, label, c)

        if self.cont_loss_fn == "cosine":
            target = torch.ones(mixed_feat.size(0), device=mixed_feat.device, dtype=torch.float32)
            cont_loss = self.contrast_loss_fn(mixed_feat, diag_desc_emb, target)  # [B] or [B,...]
        else:
            raise ValueError(f"Unsupported cont_loss: {self.cont_loss_fn}")

        if cont_loss.dim() > 1:
            cont_loss = cont_loss.mean(dim=tuple(range(1, cont_loss.dim())))  # per-sample
        cont_loss = cont_loss.mean()  # scalar over batch

        loss = (1.0 - self.loss_lambda) * surv_loss + self.loss_lambda * cont_loss
        # Ensure scalar (NLLSurvLoss should be scalar; if not, force mean)
        if loss.dim() > 0:
            loss = loss.mean()

        hazards = torch.sigmoid(logit)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        return {
            'loss': loss,                              # scalar
            'event_time': self.__to_list(event_time),
            'label': self.__to_list(label),
            'c': self.__to_list(c),
            'risk': self.__to_list(risk),
            'source': source,
            'surv_loss': surv_loss.detach(),
            'cont_loss': cont_loss.detach()
        }

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = out['loss']
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    @staticmethod
    def __to_list(x):
        return x.detach().cpu().tolist()

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.valid_step_outputs['loss'].append(float(out['loss'].detach().cpu()))
        self.valid_step_outputs['surv_loss'].append(float(out['surv_loss'].detach().cpu()))
        self.valid_step_outputs['cont_loss'].append(float(out['cont_loss'].detach().cpu()))
        self.valid_step_outputs['censorship'].extend(out['c'])
        self.valid_step_outputs['event_time'].extend(out['event_time'])
        self.valid_step_outputs['risk'].extend(out['risk'])

    def on_validation_epoch_end(self):
        loss_mean = float(torch.tensor(self.valid_step_outputs['loss']).mean().item())
        surv_loss_mean = float(torch.tensor(self.valid_step_outputs['surv_loss']).mean().item())
        cont_loss_mean = float(torch.tensor(self.valid_step_outputs['cont_loss']).mean().item())
        c_index = compute_c_index(
            censorship=self.valid_step_outputs['censorship'],
            event_time=self.valid_step_outputs['event_time'],
            risk_score=self.valid_step_outputs['risk']
        )

        self.valid_step_outputs.clear()

        self.log('valid_loss', loss_mean, prog_bar=True)
        self.log('valid_surv_loss', surv_loss_mean, prog_bar=True)
        self.log('valid_cont_loss', cont_loss_mean, prog_bar=True)
        self.log('valid_c_index', c_index, prog_bar=True)

        print(
            f"\nEpoch: {self.current_epoch}\t"
            f"Valid loss: {loss_mean:.4f}/{surv_loss_mean:.4f}/{cont_loss_mean:.4f}\t"
            f"C-index: {c_index:.4f}\n"
        )

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.test_step_outputs['loss'].append(float(out['loss'].detach().cpu()))
        self.test_step_outputs['surv_loss'].append(float(out['surv_loss'].detach().cpu()))
        self.test_step_outputs['cont_loss'].append(float(out['cont_loss'].detach().cpu()))
        self.test_step_outputs['censorship'].extend(out['c'])
        self.test_step_outputs['event_time'].extend(out['event_time'])
        self.test_step_outputs['risk'].extend(out['risk'])
        self.test_step_outputs['source'].extend(out['source'])

    def on_test_epoch_end(self):
        loss_mean = float(torch.tensor(self.test_step_outputs['loss']).mean().item())
        surv_loss_mean = float(torch.tensor(self.test_step_outputs['surv_loss']).mean().item())
        cont_loss_mean = float(torch.tensor(self.test_step_outputs['cont_loss']).mean().item())
        c_index = compute_c_index(
            censorship=self.test_step_outputs['censorship'],
            event_time=self.test_step_outputs['event_time'],
            risk_score=self.test_step_outputs['risk']
        )

        self.log('test_loss', loss_mean, prog_bar=True)
        self.log('test_surv_loss', surv_loss_mean, prog_bar=True)
        self.log('test_cont_loss', cont_loss_mean, prog_bar=True)
        self.log('test_c_index', c_index, prog_bar=True)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, foreach=False)
        return optimizer


def define_checkpoint(project_path="./weights/", filename="best_checkpoint"):
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        dirpath=project_path,
        filename=filename,
        save_top_k=1,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode="min"
    )

    return [checkpoint_callback, early_stop_callback]
