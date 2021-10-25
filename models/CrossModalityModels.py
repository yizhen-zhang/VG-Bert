import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# add cross modality attention module
class CrossAttention(nn.Module):
    def __init__(self, num_heads=8, embedding_dim=768, subspace_dim=32, relation_base=115, dropout_rate=0.1):
        super().__init__()

        self.num_attention_heads = num_heads
        self.attention_head_size = int(embedding_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.subspace_dim = subspace_dim

        self.key_visual = nn.Linear(embedding_dim, self.all_head_size, bias=False)
        self.value_visual = nn.Linear(embedding_dim, self.all_head_size)
        self.query_language = nn.Linear(embedding_dim, self.all_head_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

        self.object_transform = nn.Linear(embedding_dim, self.subspace_dim)  # linear downsampling to the relational sapce
        self.relational_matrix = nn.Parameter(torch.zeros(relation_base, subspace_dim, subspace_dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.relational_matrix)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        # x shape: (batchsize, head_num, positions, head_size)

    def forward(self, x, y, attention_mask=None):
        # x: visual feature; y: language feature
        # reshape spatial features; flatten height and width dimensions
        x = x.permute(0, 2, 3, 1)
        x_shape = x.size()  # (batchsize, height, width, channel)
        x = x.view(x.size()[0], 1, -1, x.size(3)).squeeze(1)

        # attention_input: (batchsize, positions, channel)
        mixed_query_language_layer = self.query_language(y)
        query_language = mixed_query_language_layer  # save for output
        query_language_layer = self.transpose_for_scores(mixed_query_language_layer)
        mixed_key_visual_layer = self.key_visual(x)
        key_visual_layer = self.transpose_for_scores(mixed_key_visual_layer)
        key_visual = mixed_key_visual_layer.view(*x_shape).permute(0, 3, 1, 2).contiguous()
        # calculate cross attention scores
        attention_l2v_scores = torch.matmul(query_language_layer, key_visual_layer.transpose(-1, -2))
        attention_l2v_scores = attention_l2v_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # language to visual
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(3)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_l2v_scores = attention_l2v_scores + extended_attention_mask
        # Normalize the attention scores to probabilities.
        attention_l2v_probs = nn.Softmax(dim=-1)(attention_l2v_scores)

        # value and contexual representation
        mixed_value_visual_layer = self.value_visual(x)
        value_visual_layer = self.transpose_for_scores(mixed_value_visual_layer)
        # get contextual language layer from visual stream features
        context_language_layer = torch.matmul(attention_l2v_probs, value_visual_layer)  # (batchsize, head_num, language_positions, embedding_feature)
        context_language_layer = context_language_layer.permute(0, 2, 1, 3).contiguous()  # (batchsize, positions, head_num, subspace_feature)
        new_context_language_layer_shape = context_language_layer.size()[:-2] + (self.all_head_size,)
        object_representation = context_language_layer.view(*new_context_language_layer_shape)  # (batchsize, positions, channel)

        # down sample to relational space
        object_for_relation = self.object_transform(object_representation)

        # normalizing each relational matrices
        normalized_relational_matrix = self.relational_matrix
        normalized_relational_matrix = normalized_relational_matrix.view(normalized_relational_matrix.size()[0], -1)
        normalized_relational_matrix = F.normalize(normalized_relational_matrix, dim=1, p=2)
        normalized_relational_matrix = normalized_relational_matrix.view(self.relational_matrix.size())
        normalized_relational_matrix = self.dropout(normalized_relational_matrix)

        # calculate prediction for relations
        object_for_relation = object_for_relation.unsqueeze(1)
        relation_prediction = torch.matmul(torch.matmul(object_for_relation,normalized_relational_matrix),object_for_relation.transpose(-1,-2))
        # object_for_relation = object_for_relation.squeeze()

        return query_language, key_visual, object_representation, relation_prediction, normalized_relational_matrix
