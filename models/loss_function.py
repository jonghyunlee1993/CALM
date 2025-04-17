import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    
    y = y.type(torch.int64).squeeze(0)
    c = c.type(torch.int64).squeeze(0)
    hazards = torch.sigmoid(h)
    
    S = torch.cumprod(1 - hazards, dim=1)
    
    S_padded = torch.cat([torch.ones_like(c), S], 1)

    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Initialize the InfoNCE Loss module.
        
        Args:
            temperature: Float, the temperature scaling factor.
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        """
        Forward pass for the InfoNCE loss calculation.
        
        Args:
            image_features: Tensor of shape [batch_size, feature_dim], the feature embeddings from the image encoder.
            text_features: Tensor of shape [batch_size, feature_dim], the feature embeddings from the text encoder.
        
        Returns:
            loss: Scalar Tensor, the InfoNCE loss.
        """
        # Normalize the feature vectors
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # Compute the cosine similarities between all pairs of image and text features
        logits_per_image = torch.matmul(image_features, text_features.T) / self.temperature
        logits_per_text = logits_per_image.T
        
        # Create the ground truth labels (each image corresponds to the same index text as positive pair)
        labels = torch.arange(image_features.shape[0], device=image_features.device)
        
        # Calculate the InfoNCE loss for both image-to-text and text-to-image
        loss_image_to_text = F.cross_entropy(logits_per_image, labels)
        loss_text_to_image = F.cross_entropy(logits_per_text, labels)
        
        # Return the average loss
        loss = (loss_image_to_text + loss_text_to_image) / 2
        
        return loss


class CosineEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.0):
        """
        Initializes the CosineEmbeddingLoss module.
        
        :param margin: Float, margin for dissimilarity. Should be a non-negative value.
        """
        super().__init__()
        self.margin = margin

    def forward(self, input1, input2):
        """
        Computes the cosine embedding loss between input1 and input2.
        
        :param input1: Tensor of shape [B, D], where B is the batch size and D is the embedding dimension.
        :param input2: Tensor of shape [B, D], where B is the batch size and D is the embedding dimension.
        
        :return: Scalar tensor representing the loss.
        """
        target = torch.ones(input1.size(0)).to(input1.device)
        
        cosine_similarity = F.cosine_similarity(input1, input2, dim=-1)
        loss = torch.where(target == 1, 
                           1 - cosine_similarity, 
                           torch.clamp(cosine_similarity - self.margin, min=0))
        
        return loss.mean()



if __name__ == "__main__":
    from sksurv.metrics import concordance_index_censored
    
    h = torch.rand((100, 4, 1))    
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    
    print(h.shape, hazards.shape, survival.shape, risk.shape)
    
    all_censorships = np.random.randint(0, 1, 100)
    all_event_times = np.random.randint(10, 500, 100)
    all_risk_scores = risk.reshape(-1)
    
    print(all_censorships.shape, all_event_times.shape, all_risk_scores.shape)
    
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print(c_index)