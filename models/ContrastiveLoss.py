import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F

def compute_matchmap_similarity_matrix(image_outputs, language_outputs, nwords):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes language_outputs is a (batchsize, embedding_dim, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(language_outputs.dim() == 3)
    n = image_outputs.size(0)
    T = language_outputs.size(1)
    image_outputs = image_outputs.permute(0, 2, 3, 1)
    image_outputs_shape = image_outputs.size()
    # image_outputs_shape: (batchsize, height, width, channel)
    image_outputs = image_outputs.view(image_outputs.size()[0], 1, -1, image_outputs.size(3)).squeeze(1)
    S = torch.matmul(image_outputs.unsqueeze(1), language_outputs.transpose(-1, -2))
    sim_record = torch.zeros((n,n), device=image_outputs.device)
    S,_ = S.max(2)
    for caption_idx in range(n):
        nW = max(1, nwords[caption_idx])
        sim_record[:,caption_idx] = S[:,caption_idx,0:nW].mean(1)
    return sim_record

def cross_modal_contrastive_loss(image_outputs, language_outputs, nwords, sigma=1.):
    """
    Computes the contrastive loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    sigma: temperature parameter
    """
    assert(image_outputs.dim() == 4)
    assert(language_outputs.dim() == 3)
    n = image_outputs.size(0)
    loss_I2L = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    loss_L2I = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    sim_record = compute_matchmap_similarity_matrix(image_outputs, language_outputs, nwords)
    softmax_prob = torch.nn.Softmax(dim=0)
    for i in range(n):
        n_loss_I2L = -torch.log(softmax_prob(1/sigma * sim_record[i,:])[i])
        loss_I2L = loss_I2L + n_loss_I2L
        n_loss_L2I = -torch.log(softmax_prob(1/sigma * sim_record[:,i])[i])
        loss_L2I = loss_L2I + n_loss_L2I
    loss_I2L = loss_I2L/n
    loss_L2I = loss_L2I/n
    return loss_I2L, loss_L2I

def finetune_pooled_contrastive_loss(image_outputs, language_outputs, sigma=1.):
    """
    Computes the contrastive loss for each anchor image/caption pair (pooled embedding; cosine similarity)
    The impostor image/caption is randomly sampled from the minibatch
    sigma: temperature parameter
    """
    assert(image_outputs.dim() == 2)
    assert(language_outputs.dim() == 2)
    n = image_outputs.size(0)
    # normalize to sphere for calculating cosine similarity
    image_features = F.normalize(image_outputs, dim=1)
    language_features = F.normalize(language_outputs, dim=1)
    loss_I2L = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    loss_L2I = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    sim_record = torch.matmul(image_features,language_features.T)
    softmax_prob = torch.nn.Softmax(dim=0)
    for i in range(n):
        n_loss_I2L = -torch.log(softmax_prob(1/sigma * sim_record[i,:])[i])
        loss_I2L = loss_I2L + n_loss_I2L
        n_loss_L2I = -torch.log(softmax_prob(1/sigma * sim_record[:,i])[i])
        loss_L2I = loss_L2I + n_loss_L2I
    loss_I2L = loss_I2L/n
    loss_L2I = loss_L2I/n
    return loss_I2L, loss_L2I

def relation_contrastive_loss(relation_prediction, relation_target, temperature=0.1):
    """
    Computes the margin ranking loss for each valid relation triples
    The impostor relation is randomly sampled from the minibatch
    temperature: temperature parameter
    """
    assert(relation_prediction.dim() == 4)
    assert(relation_target.dim() == 3)
    B = relation_target.size(0)
    loss_pair = torch.zeros(1, device=relation_target.device, requires_grad=True)
    loss_relation = torch.zeros(1, device=relation_target.device, requires_grad=True)
    relation_ind = relation_target.view(B,-1)  # size: Batchsize x (N_obj^2)
    relation_predflat = relation_prediction.view(B,relation_prediction.size()[1],-1)  # size: Batchsize x Class x (N_obj^2)
    softmax_prob = torch.nn.Softmax(dim=0)
    for b in range(B):
        valid_ind = torch.nonzero(relation_ind[b])  # size: 1D vector with object word pairs in an image
        relation_value = relation_predflat[b]  # size: Class x (N_obj^2)
        for k in range(len(valid_ind)):
            ij = valid_ind[k]  # the groundtruth index of k-th valid pair
            true_class = relation_ind[b,ij]
            # one direction: negative samples from false object pairs
            positive_score = relation_value[true_class,ij]
            negative_mask = torch.logical_and((relation_ind != 0),(relation_ind != true_class))
            negative_score = torch.masked_select(relation_predflat[:,true_class,:].squeeze(), negative_mask)
            score_vector = 1/temperature * torch.cat((positive_score, negative_score), dim=0)
            n_pair_loss = -torch.log(softmax_prob(score_vector)[0])
            loss_pair = loss_pair + n_pair_loss
            # the other direction: negative samples from false relation embedding
            n_relation_loss = -torch.log(softmax_prob(1/temperature * relation_value[:,ij])[true_class])  # similar to parametric approach
            loss_relation = loss_relation + n_relation_loss
            # loss = loss + n_pair_loss + n_relation_loss
    loss_pair = loss_pair / B
    loss_relation = loss_relation / B
    return loss_pair, loss_relation
