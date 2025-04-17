import os
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

class SurvivalDataset(Dataset):
    def __init__(self,
                 project_path,
                 meta_data,
                 target_project,
                 diagnostic_data="/project/kimlab_tcga/HIPPO/data/diagnostic_description.csv",
                 image_feat_path=None,
                 text_feat_path=None,
                 omic_feat_path=None,
                 model_type="BioMedBERT",
                 **kwargs):
        self.data = meta_data
        self.target_project = target_project
        self.diagnostic_data = diagnostic_data
        self.__load_diagnositc_description()
        self.model_type = model_type
        
        self.use_wsi = True if image_feat_path != None else False
        if self.use_wsi:
            self.image_feat_path = path_concat(project_path, image_feat_path)
        
        self.use_text = True if text_feat_path != None else False
        if self.use_text:
            text_feat_path = path_concat(project_path, text_feat_path)
        self.text_feat = self.__load_text_feat(text_feat_path)
        
        self.use_omic = True if omic_feat_path != None else False
        if self.use_omic:
            omic_feat_path = path_concat(project_path, omic_feat_path)
            self.omic_feat = self.__load_omic_feat(omic_feat_path)
    
    def __load_diagnositc_description(self):
        self.diagnostic_description = pd.read_csv(self.diagnostic_data).set_index("project")
        
    def __load_text_feat(self, text_feat_path):
        if text_feat_path != None:
            return pd.read_csv(text_feat_path)
        else:
            return None
    
    def __load_omic_feat(self, omic_feat_path):
        if omic_feat_path != None:
            self.use_omic = True
            return pd.read_csv(omic_feat_path)
        else:
            return None
    
    def get_wsi_feat(self, slide_id):
        return torch.load(os.path.join(self.image_feat_path, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs'))))
        
    def get_text_feat(self, case_id):
        try:
            if self.model_type == "BioMedBERT":
                text_feat = self.text_feat.loc[self.text_feat.patient_filename.isin([case_id]), "prepared_text"].values.tolist()[0]
            elif self.model_type == "LongFormer":
                text_feat = self.text_feat.loc[self.text_feat.patient_filename.isin([case_id]), "text"].values.tolist()[0]
        except:
            text_feat = "Not applicable"
        
        return text_feat
    
    def get_diagnostic_description(self, label):
        if label == 0: # severe
            diagnostic_description = self.diagnostic_description.loc[self.target_project, "high_risk"]       
        elif label == 1 or label == 2: # intermediate
            diagnostic_description = self.diagnostic_description.loc[self.target_project, "intermediate_risk"]
        elif label == 3:
            diagnostic_description = self.diagnostic_description.loc[self.target_project, "low_risk"]
        else:
            raise("Not implemeted!")
        
        return diagnostic_description
    
    def get_omic_feat(self, case_id):
        pass  
    
    def __convert_to_tensor(x, dtype="float"):
        if dtype == "float":
            return torch.tensor(x).float()
        elif dtype == "int":
            return torch.tensor(x).long()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        slide_id = self.data.loc[idx, "slide_id"]
        case_id = self.data.loc[idx, "case_id"]
        event_time = self.data.loc[idx, "survival_months"]
        label = self.data.loc[idx, "label"]
        c = self.data.loc[idx, "censorship"]
        try:
            source = self.data.loc[idx, "source"]
        except:
            source = 0
        
        image_feat = self.get_wsi_feat(slide_id)
        text_feat = self.get_text_feat(case_id)
        diagnostic_descriptions = self.get_diagnostic_description(label)
        omic_feat = None
        
        return [image_feat, text_feat, diagnostic_descriptions, omic_feat, event_time, label, c, source] 

class TrainingCollator(object):
    def __init__(self, tokenizer, number_of_instances=4096, image_dim=1024, model_type="BioMedBERT"):
        self.tokenizer = tokenizer
        self.number_of_instances = number_of_instances
        self.image_dim = image_dim
        self.model_type = model_type
    
    def __process_image_feat(self, image):
        if image.shape[1] > self.number_of_instances:
            random_index = np.random.randint(0, len(image), self.number_of_instances)
            image = image[:, random_index]
            
        return image.numpy().tolist()
    
    def __process_text_feat(self, text):
        if self.model_type == "BioMedBERT":
            text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        elif self.model_type == "LongFormer":
            text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        
        return text_token
    
    def __call__(self, batch):
        image_feat, text_feat, omic_feat, event_time, label, c, source = [], [], [], [], [], [], []
        diagnostic_descriptions = []
        
        for data in batch:
            image_feat.append(self.__process_image_feat(data[0]))
            text_feat.append(data[1])
            diagnostic_descriptions.extend(data[2])
            omic_feat = None
            event_time.append(data[4])
            label.append(data[5])
            c.append(data[6])
            source.append(data[7])
            
        image_feat = torch.Tensor(image_feat).float()
        text_feat = self.__process_text_feat(text_feat)
        diagnostic_descriptions = self.__process_text_feat(diagnostic_descriptions[0])
        omic_feat = None
        event_time = torch.Tensor(event_time)
        label = torch.Tensor(label).long()
        c = torch.Tensor(c)
    
        return image_feat, text_feat, diagnostic_descriptions, omic_feat, event_time, label, c, source

class ValidCollator(object):
    def __init__(self, tokenizer, image_dim=1024, model_type="BioMedBERT"):
        self.tokenizer = tokenizer
        self.image_dim = image_dim
        self.model_type = model_type
    
    def __process_text_feat(self, text):
        if self.model_type == "BioMedBERT":
            text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        elif self.model_type == "LongFormer":
            text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        
        return text_token
    
    def __call__(self, batch):
        image_feat, text_feat, omic_feat, event_time, label, c, source = [], [], [], [], [], [], []
        diagnostic_descriptions = []
        
        for data in batch:
            image_feat.append(data[0].numpy().tolist())
            text_feat.append(data[1])
            diagnostic_descriptions.extend(data[2])
            omic_feat = None
            event_time.append(data[4])
            label.append(data[5])
            c.append(data[6])
            source.append(data[7])
            
        image_feat = torch.Tensor(image_feat).float()
        text_feat = self.__process_text_feat(text_feat)
        diagnostic_descriptions = self.__process_text_feat(diagnostic_descriptions[0])
        omic_feat = None
        event_time = torch.Tensor(event_time)
        label = torch.Tensor(label).long()
        c = torch.Tensor(c)
    
        return image_feat, text_feat, diagnostic_descriptions, omic_feat, event_time, label, c, source

class TestCollator(object):
    def __init__(self, tokenizer, image_dim=1024, model_type="BioMedBERT"):
        self.tokenizer = tokenizer
        self.image_dim = image_dim
        self.model_type = model_type
    
    def __process_text_feat(self, text):
        if self.model_type == "BioMedBERT":
            text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        elif self.model_type == "LongFormer":
            text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        
        return text_token
    
    def __call__(self, batch):
        image_feat, text_feat, omic_feat, event_time, label, c, source = [], [], [], [], [], [], []
        diagnostic_descriptions = []
        
        for data in batch:
            image_feat.append(data[0].numpy().tolist())
            text_feat.append(data[1])
            diagnostic_descriptions.extend(data[2])
            omic_feat = None
            event_time.append(data[4])
            label.append(data[5])
            c.append(data[6])
            source.append(data[7])
            
        image_feat = torch.Tensor(image_feat).float()
        text_feat = self.__process_text_feat(text_feat)
        diagnostic_descriptions = self.__process_text_feat("Not Applicable")
        omic_feat = None
        event_time = torch.Tensor(event_time)
        label = torch.Tensor(label).long()
        c = torch.Tensor(c)
    
        return image_feat, text_feat, diagnostic_descriptions, omic_feat, event_time, label, c, source

def load_train_valid_test_split(
        project_path, 
        meta_data_path, 
        split_path,
        fold_num=0, 
        valid_size=0.1, 
        random_state=42,
        **kwargs
    ):
    df = pd.read_csv(path_concat(project_path, meta_data_path))
    
    split_path_with_fold = split_path + f"splits_{fold_num}.csv"
    
    df_split = pd.read_csv(path_concat(project_path, split_path_with_fold))
    
    train_ids = df_split.train.values
    test_ids = df_split.val.values

    train_df = df.loc[df.case_id.isin(train_ids)].reset_index(drop=True)
    test_df = df.loc[df.case_id.isin(test_ids)].reset_index(drop=True)
    
    train_df, valid_df = train_test_split(train_df, test_size=valid_size, random_state=random_state)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    
    return train_df, valid_df, test_df

def path_concat(parent, child):
    return os.path.join(parent, child)

def create_data_loaders(config, project_path, fold_num, tokenizer=None, batch_size=1, num_workers=4, use_balanced_sampler=False, 
                        diagnostic_data="/project/kimlab_tcga/HIPPO/data/diagnostic_description.csv", model_type="BioMedBERT"):
    train_df, valid_df, test_df = load_train_valid_test_split(
        project_path,
        fold_num=fold_num,
        **config['dataset']
    )

    print(f'''
    Dataset Configurations:
        Train dataset: {train_df.shape}
        Valid dataset: {valid_df.shape}
        Test dataset: {test_df.shape}
    ''')

    train_dataset = SurvivalDataset(
        project_path=project_path,
        meta_data=train_df,
        diagnostic_data=diagnostic_data,
        tokenizer=tokenizer,
        model_type=model_type,
        **config['dataset']
    )

    valid_dataset = SurvivalDataset(
        project_path=project_path,
        meta_data=valid_df,
        diagnostic_data=diagnostic_data,
        tokenizer=tokenizer,
        model_type=model_type,
        **config['dataset']
    )

    test_dataset = SurvivalDataset(
        project_path=project_path,
        meta_data=test_df,
        diagnostic_data=diagnostic_data,
        tokenizer=tokenizer,
        model_type=model_type,
        **config['dataset']
    )
    
    training_collator = TrainingCollator(tokenizer, number_of_instances=4096, image_dim=1024)
    valid_collator = ValidCollator(tokenizer)
    test_collator = TestCollator(tokenizer)
    
    if use_balanced_sampler:
        sampler = define_balanced_sampler(train_df, target_col_name="label")
        train_dataloader = DataLoader(
            train_dataset, shuffle=False, sampler=sampler, batch_size=batch_size, num_workers=num_workers, collate_fn=training_collator, persistent_workers=False, pin_memory=True, 
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, collate_fn=training_collator, persistent_workers=False, pin_memory=True, 
        )
    
    valid_dataloader = DataLoader(
        valid_dataset, shuffle=False, batch_size=1, num_workers=num_workers, collate_fn=valid_collator, persistent_workers=False, pin_memory=True, 
    )
    
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=num_workers, collate_fn=test_collator, persistent_workers=False, pin_memory=True, 
    )

    return train_dataloader, valid_dataloader, test_dataloader

def define_balanced_sampler(train_df, target_col_name="label"):
    counts = np.bincount(train_df[target_col_name])
    labels_weights = 1.0 / counts
    weights = labels_weights[train_df[target_col_name]]
    sampler = WeightedRandomSampler(weights, len(weights))

    return sampler

def create_merged_data_loaders(config, project_path, fold_num, tokenizer=None, batch_size=1, num_workers=41, use_balanced_sampler=False):    
    meta_df = pd.read_csv(path_concat(config["project_path"], config["dataset"]["meta_data_path"]))
    split_df = pd.read_csv(path_concat(config["dataset"]["split_path"], f"splits_{fold_num}.csv"))
    
    train_ids = split_df.train.values
    valid_ids = split_df.val.values
    test_ids = split_df.test.values
        
    valid_ids = valid_ids[~pd.isnull(valid_ids)]
    test_ids = test_ids[~pd.isnull(test_ids)]
    
    train_df = meta_df.loc[meta_df.case_id.isin(train_ids)].reset_index(drop=True)
    valid_df = meta_df.loc[meta_df.case_id.isin(valid_ids)].reset_index(drop=True)
    test_df = meta_df.loc[meta_df.case_id.isin(test_ids)].reset_index(drop=True)
        
    train_dataset = SurvivalDataset(
        project_path=project_path,
        meta_data=train_df,
        tokenizer=tokenizer,
        **config['dataset']
    )

    valid_dataset = SurvivalDataset(
        project_path=project_path,
        meta_data=valid_df,
        tokenizer=tokenizer,
        **config['dataset']
    )

    test_dataset = SurvivalDataset(
        project_path=project_path,
        meta_data=test_df,
        tokenizer=tokenizer,
        **config['dataset']
    )
    
    training_collator = TrainingCollator(tokenizer, number_of_instances=4096, image_dim=1024)
    test_collator = TestCollator(tokenizer)
    
    if use_balanced_sampler:
        sampler = define_balanced_sampler(train_df, target_col_name="source_index")
        train_dataloader = DataLoader(
            train_dataset, shuffle=False, sampler=sampler, batch_size=batch_size, num_workers=num_workers, collate_fn=training_collator, persistent_workers=False, pin_memory=True, 
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, collate_fn=training_collator, persistent_workers=False, pin_memory=True, 
        )
    
    valid_dataloader = DataLoader(
        valid_dataset, shuffle=False, batch_size=1, num_workers=num_workers, collate_fn=test_collator, persistent_workers=False, pin_memory=True, 
    )
    
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=num_workers, collate_fn=test_collator, persistent_workers=False, pin_memory=True, 
    )

    return train_dataloader, valid_dataloader, test_dataloader

if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    
    train_df, valid_df, test_df = load_train_valid_test_split(
        project_path="/project/kimlab_hnsc/multimodality_prognosis/HIPPO/",
        meta_data_path="/project/kimlab_hnsc/multimodality_prognosis/HIPPO/data/TCGA-HNSC_processed.csv",
        split_path="splits/tcga_hnsc_PORPOISE/",
        fold_num=4,
        valid_size=0.1, random_state=42
    )
    
    train_dataset = SurvivalDataset(
        project_path="/project/kimlab_hnsc/multimodality_prognosis/HIPPO/",
        meta_data=train_df, 
        image_feat_path="/project/kimlab_hnsc/multimodality_prognosis/CLAM_prepared/feat_ResNet",
        # text_feat_path="/project/kimlab_hnsc/multimodality_prognosis/CLAM_prepared/feat_text/prompted_embeddings.pkl",
        text_feat_path="/project/kimlab_hnsc/multimodality_prognosis/HIPPO/data/TCGA_Reports_prepared.csv",
        use_gpt=False,
        tokenizer=tokenizer,
        omic_feat_path=None,
    )
    
    print(next(iter(train_dataset)))