from .ContrastiveLoss import *
import torch
import torch.nn as nn

class TwoStreamPretrain(nn.Module):
    '''
    Pretraining on COCO dataset with cross-modal contrastive learning
    '''
    def __init__(self, image_model, language_model, embedding_dim=768, sigma=0.1):
        super(TwoStreamPretrain, self).__init__()
        self.image_model = image_model
        self.language_model = language_model
        self.image_projection_head = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1, stride=1, padding=0),
        )
        self.language_projection_head = nn.Sequential(
            nn.Linear(embedding_dim,embedding_dim),
        )
        self.sigma = sigma

    def forward(self, image_input, language_input, token_type, input_mask, nwords):  # add two input terms: relation mask and relation class
        image_output = self.image_model(image_input)
        image_metric = self.image_projection_head(image_output)
        language_output = self.language_model(language_input, token_type, input_mask)  # batch x n_object x emb_size
        language_metric = self.language_projection_head(language_output)
        # loss function
        loss_I2L, loss_L2I = cross_modal_contrastive_loss(image_metric, language_metric, nwords, sigma=self.sigma)

        return loss_I2L, loss_L2I

class RelationalGrounding(nn.Module):
    '''
    Finetuning on Visual Genome dataset for relation relation prediction
    '''
    def __init__(self, image_model, language_model, cross_attention_model, embedding_dim=768, relation_size=115, temperature=1.0):
        super(RelationalGrounding, self).__init__()
        self.image_model = image_model
        self.language_model = language_model
        self.cross_attention_model = cross_attention_model
        self.relation_size = relation_size
        self.temperature = temperature
        self.object_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(256, 56),
        )

    def forward(self, image_input, object_input_all, segments_tensors_all, input_token_mask_all, input_mask, n_object, object_target, relation_target):  # add two input terms: relation mask and relation class
        language_output = self.language_model(object_input_all, segments_tensors_all, input_token_mask_all, input_mask)  # batch x n_object x emb_size
        image_output = self.image_model(image_input)
        (query_language, key_visual, object_representation, relation_prediction, normalized_relational_matrix) = self.cross_attention_model(image_output, language_output, attention_mask=input_mask)
        object_prediction = self.object_classifier(object_representation)
        object_prediction = object_prediction.permute(0, 2, 1)
        # loss functions:
        loss_cls_object = torch.nn.CrossEntropyLoss(reduction='none')(object_prediction, object_target)  # auxilary loss for classifying visually grounded objects
        loss_pair, loss_relation = relation_contrastive_loss(relation_prediction, relation_target, temperature=self.temperature)  # contrastive loss between relation and object pairs

        return loss_cls_object, loss_pair, loss_relation, relation_target, relation_prediction, object_target, object_prediction

class TransferCrossModalRetrieval(nn.Module):
    def __init__(self, image_model, language_model, cross_attention_model, embedding_dim=768, metric_dim=768, sigma=0.1):
        super(TrasferCrossModalRetrieval, self).__init__()
        self.image_model = image_model
        self.language_model = language_model
        self.cross_attention_model = cross_attention_model
        self.image_projection_head = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(embedding_dim, metric_dim, kernel_size=1, stride=1, padding=0),
        )
        self.language_projection_head = nn.Sequential(
            nn.Linear(embedding_dim,embedding_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(embedding_dim, metric_dim),
        )
        self.sigma = sigma

    def forward(self, image_input, language_input, token_type, input_mask, nwords):  # add two input terms: relation mask and relation class
        language_output = self.language_model.language_model(language_input, token_type, input_mask)[0]  #.last_hidden_state  # batch x n_word x emb_size
        image_output = self.image_model(image_input)
        (query_language, key_visual, _, _, _, _, _) = self.cross_attention_model(image_output, language_output, attention_mask=input_mask)
        image_metric = self.image_projection_head(key_visual)
        language_metric = self.language_projection_head(query_language)
        # loss functions:
        image_pool_metric = torch.mean(image_metric,(2,3))  # pool image embedding to single vector
        language_pool_metric = torch.mean(language_metric,1)  # pool language embedding to single vector
        loss_I2L, loss_L2I = finetune_pooled_contrastive_loss(image_pool_metric, language_pool_metric, sigma=self.sigma)  #

        return image_metric, language_metric, loss_I2L, loss_L2I
