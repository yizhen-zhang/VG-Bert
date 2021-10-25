import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from collections import OrderedDict

class Bert_base(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Bert_base, self).__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.hidden_dropout_prob = dropout_rate
        config.attention_probs_dropout_prob = dropout_rate
        seed_model = BertModel(config)
        base_model = BertModel.from_pretrained('bert-base-uncased')
        # load weight from base model to seed model
        new_state_dict = OrderedDict()
        for k, v in base_model.state_dict().items():
            name = k # remove `module.`
            new_state_dict[name] = v
        # load params
        seed_model.load_state_dict(new_state_dict)
        self.language_model = seed_model

    def forward(self, x, x_type, x_mask):
        outputs = self.language_model(x, token_type_ids=x_type, attention_mask=x_mask)
        encoded_layers = outputs[0]
        return encoded_layers

class Bert_object(nn.Module):
    def __init__(self, embedding_dim=768, dropout_rate=0.1):
        super(Bert_object, self).__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.hidden_dropout_prob = dropout_rate
        config.attention_probs_dropout_prob = dropout_rate
        seed_model = BertModel(config)
        base_model = BertModel.from_pretrained('bert-base-uncased')
        base_model.to('cpu')
        # load weight from base model to seed model
        new_state_dict = OrderedDict()
        for k, v in base_model.state_dict().items():
            name = k # remove `module.`
            new_state_dict[name] = v
        # load params
        seed_model.load_state_dict(new_state_dict)
        self.language_model = seed_model
        self.embedding_dim = embedding_dim

    def forward(self, x, x_segments_tensors, x_token_mask, x_mask):
        batch_num = x.size()[0]
        obj_num = x.size()[1]
        seq_len = x.size()[2]
        feature_size = self.embedding_dim
        # flatten batch input to make each phrase (attribute + object) as independent input sequence for Bert
        x = x.view(batch_num*obj_num,seq_len)
        x_segments_tensors = x_segments_tensors.view(batch_num*obj_num,seq_len)
        x_token_mask = x_token_mask.view(batch_num*obj_num,seq_len)
        outputs = self.language_model(input_ids=x, token_type_ids=x_segments_tensors, attention_mask=x_token_mask)
        # pooling
        outputs = outputs[0] * x_token_mask.unsqueeze(-1)
        outputs = torch.mean(outputs,dim=1)  # pool language representation as (attribute + object)
        outputs = outputs.view(batch_num, obj_num, feature_size)
        outputs = outputs * x_mask.unsqueeze(-1)
        return outputs
