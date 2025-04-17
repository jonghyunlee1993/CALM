import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import pandas as pd

from models.loss_function import NLLSurvLoss, InfoNCELoss, CosineEmbeddingLoss
from utils.eval_util import compute_c_index

from collections import defaultdict

class ModelInterface(pl.LightningModule):
    def __init__(self, 
                 model,
                 learning_rate,
                 cont_loss="cosine",
                 loss_lambda=0.1,
                 **kwargs):
        super().__init__()
        self.model = model
        self.surv_loss_fn = NLLSurvLoss()
        self.cont_loss = cont_loss
        
        if self.cont_loss == "cosine":
            self.contrast_loss_fn = CosineEmbeddingLoss()
        elif self.cont_loss == "mse":
            self.contrast_loss_fn = torch.nn.MSELoss()
        self.loss_lambda = loss_lambda
        
        self.learning_rate = float(learning_rate)
        self.max_epochs = kwargs.get('max_epochs', 100)  # Default to 100 if not provided
        self.warmup_epochs = kwargs.get('warmup_epochs', 10)  # Default to 10 if not provided
        
        self.valid_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)

    def forward(self, batch):
        image_feat, text_feat, diagnostic_description, omic_feat, event_time, label, c, source = batch
        logit, mixed_feat, diagnostic_description, image_CLS, text_CLS, _, _ = self.model(image_feat, text_feat, diagnostic_description)
        
        if logit.shape[0] == 1: # if batch size is 1
            label = label.unsqueeze(0)
            c = c.unsqueeze(0)
            
        surv_loss = self.surv_loss_fn(logit, label, c)
        if self.cont_loss == "cosine":
            cont_loss = self.contrast_loss_fn(mixed_feat, diagnostic_description)
        elif self.cont_loss == "mse":
            cont_loss = self.contrast_loss_fn(image_CLS, text_CLS)
        loss = (1 - self.loss_lambda) * surv_loss + self.loss_lambda * cont_loss
        
        hazards = torch.sigmoid(logit)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)
        
        return {'loss': loss, 'event_time': self.__convert_tensor_to_list(event_time), 'label': label[0], 
                'c': self.__convert_tensor_to_list(c), 'risk': self.__convert_tensor_to_list(risk), 'source': source,
                'surv_loss': surv_loss, 'cont_loss': cont_loss}

    def training_step(self, batch, batch_idx):
        result_dict = self(batch)   
        loss = result_dict['loss'].unsqueeze(0)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
        return loss

    def __convert_tensor_to_list(self, x):
        return x.detach().to("cpu").numpy().tolist()
    
    def validation_step(self, batch, batch_idx):
        result_dict = self(batch)

        self.valid_step_outputs['loss'].append(self.__convert_tensor_to_list(result_dict['loss']))
        self.valid_step_outputs['surv_loss'].append(self.__convert_tensor_to_list(result_dict['surv_loss']))
        self.valid_step_outputs['cont_loss'].append(self.__convert_tensor_to_list(result_dict['cont_loss']))
        self.valid_step_outputs['censorship'].extend(result_dict['c'])
        self.valid_step_outputs['event_time'].extend(result_dict['event_time'])
        self.valid_step_outputs['risk'].extend(result_dict['risk'])

    def on_validation_epoch_end(self):
        all_loss = torch.mean(torch.Tensor(self.valid_step_outputs['loss']))
        all_surv_loss = torch.mean(torch.Tensor(self.valid_step_outputs['surv_loss']))
        all_cont_loss = torch.mean(torch.Tensor(self.valid_step_outputs['cont_loss']))
        
        all_c_index = compute_c_index(
            censorship=self.valid_step_outputs['censorship'], 
            event_time=self.valid_step_outputs['event_time'], 
            risk_score=self.valid_step_outputs['risk']
        )
        
        self.valid_step_outputs['loss'].clear()
        self.valid_step_outputs['surv_loss'].clear()
        self.valid_step_outputs['cont_loss'].clear()
        self.valid_step_outputs['censorship'].clear()
        self.valid_step_outputs['event_time'].clear()
        self.valid_step_outputs['risk'].clear()
        
        self.log('valid_loss', all_loss, prog_bar=True)
        self.log('valid_surv_loss', all_surv_loss, prog_bar=True)
        self.log('valid_cont_loss', all_cont_loss, prog_bar=True)
        self.log('valid_c_index', all_c_index, prog_bar=True)

        print(f"\n\nEpoch: {self.current_epoch}\tValid loss: {round(all_loss.detach().to('cpu').numpy().tolist(), 4)}/{round(all_surv_loss.detach().to('cpu').numpy().tolist(), 4)}/{round(all_cont_loss.detach().to('cpu').numpy().tolist(), 4)}\tC-index: {round(all_c_index, 4)}\n\n")
        
    def test_step(self, batch, batch_idx):
        result_dict = self(batch)
               
        self.test_step_outputs['loss'].append(self.__convert_tensor_to_list(result_dict['loss']))
        self.valid_step_outputs['surv_loss'].append(self.__convert_tensor_to_list(result_dict['surv_loss']))
        self.valid_step_outputs['cont_loss'].append(self.__convert_tensor_to_list(result_dict['cont_loss']))
        self.test_step_outputs['censorship'].extend(result_dict['c'])
        self.test_step_outputs['event_time'].extend(result_dict['event_time'])
        self.test_step_outputs['risk'].extend(result_dict['risk'])
        self.test_step_outputs['source'].extend(result_dict['source'])

    def on_test_epoch_end(self):
        all_loss = torch.mean(torch.Tensor(self.test_step_outputs['loss']))
        all_surv_loss = torch.mean(torch.Tensor(self.test_step_outputs['surv_loss']))
        all_cont_loss = torch.mean(torch.Tensor(self.test_step_outputs['cont_loss']))
        
        all_c_index = compute_c_index(
            censorship=self.test_step_outputs['censorship'], 
            event_time=self.test_step_outputs['event_time'], 
            risk_score=self.test_step_outputs['risk']
        )
        
        self.log('test_loss', all_loss, prog_bar=True)
        self.log('test_surv_loss', all_surv_loss, prog_bar=True)
        self.log('test_cont_loss', all_cont_loss, prog_bar=True)
        self.log('test_c_index', all_c_index, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, foreach=False)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.max_epochs)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


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
        min_delta=0.05, 
        patience=10, 
        verbose=False, 
        mode="min"
    )
    
    return [checkpoint_callback, early_stop_callback]


if __name__ == "__main__":
    pass