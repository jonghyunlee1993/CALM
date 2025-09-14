import os
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from musk import utils as musk_utils


class SurvivalDataset(Dataset):
    def __init__(
        self,
        project_path,
        meta_data,
        target_project,
        anchor_data="/project/kimlab_tcga/JH_workspace/multimodality_prognosis_prediction/CALM/data/anchot_text/anchor_text_short_context.csv",
        image_feat_path=None,
        text_feat_path=None,
        text_encoder_type="musk",
        **kwargs
    ):
        self.data = meta_data
        self.target_project = target_project
        self.anchor_data = anchor_data
        self.__load_diagnositc_description()
        self.text_encoder_type = text_encoder_type

        self.use_wsi = True if image_feat_path is not None else False
        if self.use_wsi:
            self.image_feat_path = path_concat(project_path, image_feat_path)

        self.use_text = True if text_feat_path is not None else False
        if self.use_text:
            text_feat_path = path_concat(project_path, text_feat_path)

        self.text_reports = self.__load_text_feat(text_feat_path)

    def __load_diagnositc_description(self):
        self.diagnostic_description = pd.read_csv(self.anchor_data).set_index("project")

    def __load_text_feat(self, text_feat_path):
        if text_feat_path is not None:
            return pd.read_csv(text_feat_path)
        else:
            return None

    def get_wsi_feat(self, slide_id):
        return torch.load(os.path.join(self.image_feat_path, f'{slide_id.rstrip(".svs")}.pt'))

    def get_text_feat(self, case_id):
        try:
            if self.text_encoder_type == "bert":
                text_feat = self.text_reports.loc[
                    self.text_reports.patient_id.isin([case_id]), "structured_summary_long_context"
                ].values.tolist()[0]
            elif self.text_encoder_type == "musk":
                text_feat = self.text_reports.loc[
                    self.text_reports.patient_id.isin([case_id]), "structured_summary_short_context"
                ].values.tolist()[0]
            else:
                text_feat = "Text report is not available."
        except Exception:
            text_feat = "Text report is not available."
        return text_feat

    def get_diagnostic_description(self, label):
        if label == 0:
            dd = self.diagnostic_description.loc[self.target_project, "high_risk"]
        elif label == 1 or label == 2:
            dd = self.diagnostic_description.loc[self.target_project, "intermediate_risk"]
        elif label == 3:
            dd = self.diagnostic_description.loc[self.target_project, "low_risk"]
        else:
            raise RuntimeError("Unsupported label")
        return dd

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
        except Exception:
            source = 0

        image_feat = self.get_wsi_feat(slide_id)
        text_feat = self.get_text_feat(case_id)
        diagnostic_descriptions = self.get_diagnostic_description(label)
        omic_feat = None

        return [image_feat, text_feat, diagnostic_descriptions, omic_feat, event_time, label, c, source]


class TrainingCollator(object):
    def __init__(self, tokenizer, number_of_instances=4096, image_dim=1024, text_encoder_type="musk"):
        self.tokenizer = tokenizer
        self.number_of_instances = number_of_instances
        self.image_dim = image_dim
        self.text_encoder_type = text_encoder_type

    def __process_image_feat(self, image):
        # image: [num_instances, feat_dim]
        if image.shape[0] > self.number_of_instances:  # fixed: was shape[1]
            random_index = np.random.choice(image.shape[0], self.number_of_instances, replace=False)
            image = image[random_index, :]
        return image.cpu().numpy().tolist() if isinstance(image, torch.Tensor) else image

    def __process_text_feat(self, text):
        if self.text_encoder_type == "bert":
            text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            return text_token
        elif self.text_encoder_type == "musk":
            tokens = []
            for t in text:
                text_token, _ = musk_utils.xlm_tokenizer(t, self.tokenizer, max_len=100)
                if isinstance(text_token, list):
                    text_token = torch.tensor(text_token, dtype=torch.long)
                tokens.append(text_token)
            max_len = max(token.size(0) for token in tokens)
            padded_tokens = []
            for token in tokens:
                if token.size(0) < max_len:
                    padding = torch.zeros(max_len - token.size(0), dtype=torch.long)
                    token = torch.cat([token, padding])
                padded_tokens.append(token)
            return torch.stack(padded_tokens)

    def __call__(self, batch):
        image_feat, texts, event_time, label, c, source = [], [], [], [], [], []
        diagnostic_descriptions = []

        for data in batch:
            image_feat.append(self.__process_image_feat(data[0]))
            texts.append(data[1])
            diagnostic_descriptions.append(data[2])
            event_time.append(data[4]); label.append(data[5]); c.append(data[6]); source.append(data[7])

        image_feat = torch.tensor(image_feat, dtype=torch.float32)
        text_feat = self.__process_text_feat(texts)
        diagnostic_descriptions = self.__process_text_feat(diagnostic_descriptions)

        omic_feat = None
        event_time = torch.tensor(event_time, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        c = torch.tensor(c, dtype=torch.float32)

        return image_feat, text_feat, diagnostic_descriptions, omic_feat, event_time, label, c, source


class ValidCollator(object):
    def __init__(self, tokenizer, image_dim=1024, text_encoder_type="musk"):
        self.tokenizer = tokenizer
        self.image_dim = image_dim
        self.text_encoder_type = text_encoder_type

    def __process_text_feat(self, text):
        if self.text_encoder_type == "bert":
            if isinstance(text, list):
                text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            else:
                text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            return text_token
        elif self.text_encoder_type == "musk":
            if isinstance(text, list):
                tokens = []
                for t in text:
                    text_token, _ = musk_utils.xlm_tokenizer(t, self.tokenizer, max_len=100)
                    if isinstance(text_token, list):
                        text_token = torch.tensor(text_token, dtype=torch.long)
                    tokens.append(text_token)
                max_len = max(token.size(0) for token in tokens)
                padded_tokens = []
                for token in tokens:
                    if token.size(0) < max_len:
                        padding = torch.zeros(max_len - token.size(0), dtype=torch.long)
                        token = torch.cat([token, padding])
                    padded_tokens.append(token)
                return torch.stack(padded_tokens)
            else:
                text_token, _ = musk_utils.xlm_tokenizer(text, self.tokenizer, max_len=100)
                if isinstance(text_token, list):
                    text_token = torch.tensor(text_token, dtype=torch.long)
                if text_token.dim() == 1:
                    text_token = text_token.unsqueeze(0)
                return text_token
        return text_token

    def __call__(self, batch):
        image_feat, texts, event_time, label, c, source = [], [], [], [], [], []
        diagnostic_descriptions = []

        for data in batch:
            image_feat.append(data[0].cpu().numpy().tolist() if isinstance(data[0], torch.Tensor) else data[0])
            texts.append(data[1])
            diagnostic_descriptions.append(data[2])
            event_time.append(data[4]); label.append(data[5]); c.append(data[6]); source.append(data[7])

        image_feat = torch.tensor(image_feat, dtype=torch.float32)
        text_feat = self.__process_text_feat(texts)
        diagnostic_descriptions = self.__process_text_feat(diagnostic_descriptions)

        omic_feat = None
        event_time = torch.tensor(event_time, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        c = torch.tensor(c, dtype=torch.float32)

        return image_feat, text_feat, diagnostic_descriptions, omic_feat, event_time, label, c, source


class TestCollator(object):
    def __init__(self, tokenizer, image_dim=1024, text_encoder_type="musk"):
        self.tokenizer = tokenizer
        self.image_dim = image_dim
        self.text_encoder_type = text_encoder_type

    def __process_text_feat(self, text):
        if self.text_encoder_type == "bert":
            if isinstance(text, list):
                text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            else:
                text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            return text_token
        elif self.text_encoder_type == "musk":
            if isinstance(text, list):
                tokens = []
                for t in text:
                    text_token, _ = musk_utils.xlm_tokenizer(t, self.tokenizer, max_len=100)
                    if isinstance(text_token, list):
                        text_token = torch.tensor(text_token, dtype=torch.long)
                    tokens.append(text_token)
                max_len = max(token.size(0) for token in tokens)
                padded_tokens = []
                for token in tokens:
                    if token.size(0) < max_len:
                        padding = torch.zeros(max_len - token.size(0), dtype=torch.long)
                        token = torch.cat([token, padding])
                    padded_tokens.append(token)
                return torch.stack(padded_tokens)
            else:
                text_token, _ = musk_utils.xlm_tokenizer(text, self.tokenizer, max_len=100)
                if isinstance(text_token, list):
                    text_token = torch.tensor(text_token, dtype=torch.long)
                if text_token.dim() == 1:
                    text_token = text_token.unsqueeze(0)
                return text_token
        return text_token

    def __call__(self, batch):
        image_feat, texts, event_time, label, c, source = [], [], [], [], [], []
        diagnostic_descriptions = []

        for data in batch:
            image_feat.append(data[0].cpu().numpy().tolist() if isinstance(data[0], torch.Tensor) else data[0])
            texts.append(data[1])
            diagnostic_descriptions.append("Not available.")
            event_time.append(data[4]); label.append(data[5]); c.append(data[6]); source.append(data[7])

        image_feat = torch.tensor(image_feat, dtype=torch.float32)
        text_feat = self.__process_text_feat(texts)
        diagnostic_descriptions = self.__process_text_feat(diagnostic_descriptions)

        omic_feat = None
        event_time = torch.tensor(event_time, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        c = torch.tensor(c, dtype=torch.float32)

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
    split_path_with_fold = split_path + f"refined_splits_{fold_num}.csv"
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


def create_data_loaders(
    config,
    project_path,
    fold_num,
    tokenizer=None,
    batch_size=1,
    num_workers=4,
    use_balanced_sampler=False,
    anchor_data="/project/kimlab_tcga/JH_workspace/multimodality_prognosis_prediction/CALM/data/anchot_text/anchor_text_short_context.csv",
    text_encoder_type="musk"
):
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
        anchor_data=anchor_data,
        tokenizer=tokenizer,
        text_encoder_type=text_encoder_type,
        **config['dataset']
    )

    valid_dataset = SurvivalDataset(
        project_path=project_path,
        meta_data=valid_df,
        anchor_data=anchor_data,
        tokenizer=tokenizer,
        text_encoder_type=text_encoder_type,
        **config['dataset']
    )

    test_dataset = SurvivalDataset(
        project_path=project_path,
        meta_data=test_df,
        anchor_data=anchor_data,
        tokenizer=tokenizer,
        text_encoder_type=text_encoder_type,
        **config['dataset']
    )

    training_collator = TrainingCollator(
        tokenizer, number_of_instances=4096, image_dim=1024, text_encoder_type=text_encoder_type
    )
    valid_collator = ValidCollator(tokenizer, text_encoder_type=text_encoder_type)
    test_collator = TestCollator(tokenizer, text_encoder_type=text_encoder_type)

    if use_balanced_sampler:
        sampler = define_balanced_sampler(train_df, target_col_name="label")
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=False,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=training_collator,
            persistent_workers=False,
            pin_memory=True,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=training_collator,
            persistent_workers=False,
            pin_memory=True,
        )

    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=valid_collator,
        persistent_workers=False,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=test_collator,
        persistent_workers=False,
        pin_memory=True,
    )

    return train_dataloader, valid_dataloader, test_dataloader


def define_balanced_sampler(train_df, target_col_name="label"):
    counts = np.bincount(train_df[target_col_name])
    labels_weights = 1.0 / counts
    weights = labels_weights[train_df[target_col_name]]
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler
