import string
import random
import argparse
from models.encoders.AttentionMIL import GatedAttentionMIL
from models.model import MMEncoder, TextEncoder
from utils.config_util import read_config
from utils.data_util import create_data_loaders, create_merged_data_loaders
from utils.logging_util import make_results_dir, logging
from torch.utils.data import DataLoader
from models.model_interface import ModelInterface, define_checkpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="specify config file")
    parser.add_argument('--results_path', default="results")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_sampler', default=False, type=bool)
    parser.add_argument('--image_encoder', default="ABMIL")
    parser.add_argument('--diagnostic_data', default="/project/kimlab_tcga/JH_workspace/multimodality_prognosis_prediction/CALM/data/diagnostic_description.csv")
    parser.add_argument('--cont_loss', default="cosine", choices=["cosine", "mse"])
    parser.add_argument('--loss_lambda', default=0.1, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    
    args = parser.parse_args()
    
    make_results_dir(f"{args.results_path}")
    config = read_config(args.config)
    config['trainer_param']['max_epochs'] = config['hyperparam']['max_epochs']
    project_path = config['project_path']
    
    for fold in range(5):
        print(f"fold NUMBER: {fold}")
        
        arb_str = ''.join([random.choice(string.ascii_lowercase) for _ in range(8)])
        arb_digit = ''.join(random.choices(string.digits, k=8))
        
        logger = CSVLogger(
            save_dir="logs", 
            name="CALM_image_text",
            version=arb_str + arb_digit
        )
    
        PROJECT_NAME = args.config.split("/")[-1].split(".")[0][7:]
        
        if args.image_encoder == "ABMIL":
            image_encoder = GatedAttentionMIL()
            args.is_CLS = False
        else:
            print("Not implemented yet")
            raise
        
        text_encoder = TextEncoder()
        train_dataloader, valid_dataloader, test_dataloader = create_data_loaders(config, 
                                                                                  project_path, 
                                                                                  fold_num=fold, 
                                                                                  tokenizer=text_encoder.tokenizer, 
                                                                                  batch_size=args.batch_size, 
                                                                                  num_workers=args.num_workers, 
                                                                                  use_balanced_sampler=args.use_sampler,
                                                                                  diagnostic_data=args.diagnostic_data
                                                                                 )

        model = MMEncoder(image_encoder=image_encoder, text_encoder=text_encoder, is_CLS=args.is_CLS, temperature=args.temperature)
        model_interface = ModelInterface(model=model, cont_loss=args.cont_loss, **config['hyperparam'])
        callbacks = define_checkpoint(filename=PROJECT_NAME + f"_Fold-{fold}" + f"_{arb_str + arb_digit}")
        
        trainer = Trainer(**config['trainer_param'],
                          callbacks=callbacks, 
                          logger=logger)

        trainer.fit(model_interface, train_dataloader, valid_dataloader)
        test_output = trainer.test(model_interface, test_dataloader, ckpt_path="best")
        logging(PROJECT_NAME, fold, test_output, args.results_path)