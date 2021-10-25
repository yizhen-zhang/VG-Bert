import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
import math
import numpy as np

class VGG16(nn.Module):
    def __init__(self, embedding_dim=768, pretrained=False):
        super(VGG16, self).__init__()
        # pretrained_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        seed_model = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        seed_model = nn.Sequential(*list(seed_model.children())[:-1]) # remove final maxpool
        last_layer_index = len(list(seed_model.children()))
        # add a linear embedder to match the dimension in Bert-base model
        seed_model.add_module(str(last_layer_index),
            nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0))
        self.image_model = seed_model

    def forward(self, x):
        x = self.image_model(x)
        return x

class VGG16_classification(nn.Module):
    def __init__(self, embedding_dim=768, pretrained=False):
        super(VGG16_classification, self).__init__()
        seed_model = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        seed_model = nn.Sequential(*list(seed_model.children())[:-1]) # remove final maxpool
        self.image_model = seed_model  # features
        # add a linear embedder to match the dimension in Bert-base model
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.visual_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.visual_avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(768 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.image_model(x)
        x = self.embedder(x)
        x = self.visual_maxpool(x)
        x = self.visual_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGG16_Attention(nn.Module):
    def __init__(self, embedding_dim=768, pretrained=False, use_position=None, dropout_rate=0.1):
        super(VGG16_Attention, self).__init__()
        seed_model = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        seed_model = nn.Sequential(*list(seed_model.children())[:-1]) # remove final maxpool
        self.image_model = seed_model  # features
        # add a linear embedder to match the dimension in Bert-base model
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.attention = SpatialSelfAttention(use_position=use_position, dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.image_model(x)
        x = self.embedder(x)
        x = self.attention(x)
        return x

class VGG16_Attention_classification(nn.Module):
    def __init__(self, embedding_dim=768, pretrained=False, use_position=None):
        super(VGG16_Attention_classification, self).__init__()
        seed_model = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        seed_model = nn.Sequential(*list(seed_model.children())[:-1]) # remove final maxpool
        self.image_model = seed_model  # features
        # add a linear embedder to match the dimension in Bert-base model
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.attention = SpatialSelfAttention(use_position=use_position)

        self.visual_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.visual_avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(768 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.image_model(x)
        x = self.embedder(x)
        x = self.attention(x)

        x = self.visual_maxpool(x)
        x = self.visual_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# add spatial self attention module
class SpatialSelfAttention(nn.Module):
    def __init__(self, num_heads=12, feature_map_size=196, hidden_size=768, intermediate_size=3072, use_position=None, dropout_rate=0.1):
        super().__init__()
        self.use_position = use_position
        if self.use_position=="predefine":
            self.position_encoder = PositionalEncoder(1)
        elif self.use_position=="learn":
            self.position_embeddings = nn.Embedding(feature_map_size, hidden_size)
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Linear(hidden_size, intermediate_size)
        self.ffn_act = nn.ReLU()
        self.ffn_output = nn.Linear(intermediate_size, hidden_size)
        self.full_layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        # x shape: (batchsize, head_num, positions, subspace_feature)

    def forward(self, x):
        device = x.device
        # reshape spatial features; flatten height and width dimensions
        x = x.permute(0, 2, 3, 1)
        x_shape = x.size()  # (batchsize, height, width, channel)
        x = x.view(x.size()[0], 1, -1, x.size(3)).squeeze(1)  # (batchsize, positions, channel)

        # assign position embedding
        if self.use_position is None:
            print("position encoding not defined")
        elif self.use_position=="no":  # no positional encoder
            attention_input = x
        elif self.use_position=="predefine":
            attention_input = self.position_encoder(x)
        elif self.use_position=="learn":
            position_ids = torch.arange(x.size()[1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(x.size()[:-1])
            position_encoding = self.position_embeddings(position_ids)
            attention_input = x + position_encoding
        # attention_input: (batchsize, positions, channel)

        mixed_query_layer = self.query(attention_input)
        mixed_key_layer = self.key(attention_input)
        mixed_value_layer = self.value(attention_input)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # *_layer: (batchsize, head_num, positions, subspace_feature)

        if self.use_position=="relative_learn":
            position_layer = self.transpose_for_scores(relative_position_encoding)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) + torch.matmul(query_layer, position_layer.transpose(-1, -2))
        else:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)  # (batchsize, head_num, positions, subspace_feature)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (batchsize, positions, head_num, subspace_feature)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # (batchsize, positions, channel)

        # output layer
        context_layer = self.dense(context_layer)
        context_layer = self.dropout(context_layer)
        context_layer = self.LayerNorm(context_layer + x)

        # intermediate linear layer
        hidden_states = self.ffn(context_layer)
        hidden_states = self.ffn_act(hidden_states)
        hidden_states = self.ffn_output(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.full_layernorm(hidden_states + context_layer)
        hidden_states = hidden_states.view(*x_shape).permute(0, 3, 1, 2).contiguous()

        return hidden_states

class PositionalEncoder(nn.Module):
    def __init__(self, beta, d_model=int(768/2), d_map=14):
        super().__init__()
        self.d_model = d_model
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = np.zeros((d_map,d_model))
        for pos in range(d_map):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        PE = np.zeros((d_map,d_map,d_model*2))
        for x in range(d_map):
            for y in range(d_map):
                PE[x,y,0:d_model] = pe[x,:]
                PE[x,y,d_model:] = pe[y,:]
        PE = PE.reshape(-1, PE.shape[-1])
        PE = torch.from_numpy(PE).float()
        PE = PE.unsqueeze(0)
        self.register_buffer('PE', PE)
        self.beta = beta

    def forward(self, x):
        device = x.device
        #add constant to embedding
        position_encoding = self.PE
        position_encoding = position_encoding.to(device)
        position_encoding.requires_grad=False
        x = x + self.beta*position_encoding
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, padding_mode='replicate')
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}
