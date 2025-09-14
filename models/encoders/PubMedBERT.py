import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class PubMedBERT(nn.Module):
    def __init__(self, layer_index=12, text_out_dim=512, **kwargs):
        super().__init__()
        self.model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.__freeze_weights(layer_index=layer_index)
        self.pooler = nn.Sequential(
            nn.Linear(768, text_out_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
    
    def __freeze_weights(self, layer_index=12):
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

        for layer in self.model.encoder.layer[:layer_index]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, **x):
        outputs = self.model(**x)
        hidden_state = outputs['last_hidden_state']
        cls_out = self.pooler(hidden_state[:, 0])
        
        return hidden_state, cls_out
        
        
if __name__ == "__main__":
    pubmed_bert = PubMedBERT(layer_index=12)
    
    input_text_short = "This is test inputs for digital pathology reports. "
    input_text_long = """
    This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. 
    This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. 
    This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. 
    This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. This is test inputs for digital pathology reports.This is test inputs for digital pathology reports. 
    """
        
    tokenized_text = pubmed_bert.tokenizer(input_text_long, return_tensors="pt", truncation=True, max_length=512)
    
    print(f"Tokenized text: {tokenized_text}")
    
    import torch
    with torch.no_grad():
        hidden_state, cls_out = pubmed_bert(**tokenized_text)
    
    print(f"Model cls_out: {cls_out.shape}")
    
    