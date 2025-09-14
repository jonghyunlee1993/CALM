import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import string
import random
import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from models.encoders.AttentionMIL import GatedAttentionMIL
from models.model import MMEncoder, TextEncoder
from models.model_interface import ModelInterface, define_checkpoint

from utils.config_util import read_config
from utils.data_util import create_data_loaders
from utils.logging_util import make_results_dir, logging

from transformers import BertTokenizer
from transformers import XLMRobertaTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="specify config file", required=True)
    args = parser.parse_args()

    config = read_config(args.config)
    config['trainer_param']['max_epochs'] = config['hyperparam']['max_epochs']

    project_path = config['project_path']
    training_config = config['training']
    results_path = training_config['results_path']
    batch_size = training_config['batch_size']
    num_workers = training_config['num_workers']
    use_sampler = training_config['use_sampler']
    cont_loss = training_config['cont_loss']
    loss_lambda = training_config['loss_lambda']
    image_encoder_type = training_config['image_encoder']
    text_encoder_type = training_config['text_encoder']

    make_results_dir(f"{results_path}")

    if config['training']['fold'] == -1:
        folds = range(5)
    else:
        folds = [config['training']['fold']]

    for fold in folds:
        print(f"fold NUMBER: {fold}")

        arb_str = ''.join([random.choice(string.ascii_lowercase) for _ in range(8)])
        arb_digit = ''.join(random.choices(string.digits, k=8))

        logger = CSVLogger(
            save_dir="logs",
            name="CALM_image_text",
            version=arb_str + arb_digit
        )

        PROJECT_NAME = args.config.split("/")[-1].split(".")[0][7:]

        image_input_dim = 1024
        image_encoder = GatedAttentionMIL(L=image_input_dim)

        text_encoder = TextEncoder(text_encoder_type=text_encoder_type)
        if text_encoder_type == "musk":
            import os
            print(f"Tokenizer file availability: {os.path.exists('./MUSK/musk/models/tokenizer.spm')}")
            
            tokenizer = XLMRobertaTokenizer.from_pretrained("./MUSK/musk/models/tokenizer.spm")
        elif text_encoder_type == "bert":
            tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        else:
            raise NotImplementedError(f"Text encoder {text_encoder_type} not implemented")

        train_dataloader, valid_dataloader, test_dataloader = create_data_loaders(
            config,
            project_path,
            fold_num=fold,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            use_balanced_sampler=use_sampler,
            text_encoder_type=text_encoder_type
        )

        model = MMEncoder(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_encoder_type=text_encoder_type
        )

        model_interface = ModelInterface(
            model=model,
            cont_loss=cont_loss,
            loss_lambda=loss_lambda,
            **config['hyperparam']
        )

        callbacks = define_checkpoint(filename=PROJECT_NAME + f"_Fold-{fold}" + f"_{arb_str + arb_digit}")

        trainer = Trainer(
            **config['trainer_param'],
            callbacks=callbacks,
            gradient_clip_val=1.0,
            logger=logger,
            log_every_n_steps=32
        )

        trainer.fit(model_interface, train_dataloader, valid_dataloader)
        test_output = trainer.test(model_interface, test_dataloader, ckpt_path="best")
        logging(PROJECT_NAME, fold, test_output, results_path)
