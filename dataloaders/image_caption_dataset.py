import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
import numpy as np
import os
from PIL import Image
from transformers import BertTokenizer, BertModel, BertForMaskedLM

class LoadVisualGenomeDataset(Dataset):
    def __init__(self, dataset_json_file, image_conf=None, relation_num = 115):
        """
        Dataset that manages a set of paired images and object-relation lists in the Visual Genome dataset

        :param dataset_json_file
        :param image_conf: Dictionary containing 'crop_size' and 'center_crop'
        :param relation_num: number of relation labels - default=115 after filtering the dataset
        """
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.image_base_path = data_json['image_base_path']
        self.relation_num = relation_num

        if not image_conf:
            self.image_conf = {}
        else:
            self.image_conf = image_conf

        center_crop = self.image_conf.get('center_crop', False)
        crop_size = self.image_conf.get('crop_size', 224)

        if center_crop:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        else:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])

        RGB_mean = self.image_conf.get('RGB_mean', [0.485, 0.456, 0.406])
        RGB_std = self.image_conf.get('RGB_std', [0.229, 0.224, 0.225])
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

        # tokenizer for Bert model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def _LoadObject(self, object_name_list, object_class_list, relation_list):
        """
        returns:
            object_input_all: (max_num_object, max_num_tokens) tokenized object descriptions
            segments_tensors_all: (max_num_object, max_num_tokens) all 1s
            input_token_mask_all: (max_num_object, max_num_tokens) 1 for valid tokens; 0 for padding tokens
            input_mask: (max_num_object,) 1 for valid objects; 0 for padding objects
            n_object: int; number of valid objects
            object_target: (max_num_object,) ground truth label for object class; 0 for padding objects
            relation_target: (max_num_object, max_num_object) ground truth label for pari-wise relation; 0 for padding object pairs
        """
        # curriculum learning
        relation_num = self.relation_num
        # load tokens in captions
        max_num_object = 32  # each image with at most 32 objects
        max_num_tokens = 10  # each object with at most 10 tokens in its description
        object_input_all = torch.zeros(max_num_object, max_num_tokens).type('torch.LongTensor')  # (max_num_object, max_num_tokens)
        segments_tensors_all = torch.zeros(max_num_object, max_num_tokens).type('torch.LongTensor')  # (max_num_object, max_num_tokens)
        input_token_mask_all = torch.zeros(max_num_object, max_num_tokens).type('torch.LongTensor')  # (max_num_object, max_num_tokens)
        object_target = torch.zeros(max_num_object).type('torch.LongTensor')  # (num_object,)
        relation_target = torch.zeros(max_num_object, max_num_object).type('torch.LongTensor')  # (num_object, num_object)
        relation_list = np.asarray(relation_list)
        # create object number mask
        n_object = len(object_name_list)
        input_mask = np.ones(n_object)
        p = max_num_object - n_object
        if p > 0:
            input_mask = np.pad(input_mask, (0, p), 'constant', constant_values=(0, 0))
        elif p < 0:
            input_mask = input_mask[0:p]
            n_object = max_num_object
        input_mask = torch.from_numpy(input_mask)
        input_mask = input_mask.type('torch.LongTensor')
        # create labels for object and relation
        if(n_object>0):
            object_target[0:n_object] = torch.from_numpy(np.array(object_class_list))[0:n_object]
            relation_target[0:n_object, 0:n_object] = torch.from_numpy(np.array(relation_list))[0:n_object, 0:n_object]
        # create input for each object
        for i in range(n_object):
            caption = '[CLS] ' + object_name_list[i] + ' [SEP]'
            tokenized_text = self.tokenizer.tokenize(caption)
            nw = len(tokenized_text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            input_token_mask = np.ones(len(tokenized_text))
            # cut sequence if too long
            p = max_num_tokens - nw
            if p > 0:
                indexed_tokens = np.pad(indexed_tokens, (0, p), 'constant', constant_values=(0, 0))
                input_token_mask = np.pad(input_token_mask, (0, p), 'constant', constant_values=(0, 0))
            elif p < 0:
                indexed_tokens = indexed_tokens[0:p]
                input_token_mask = input_token_mask[0:p]
                nw = max_num_tokens
            tokens_tensor = torch.tensor([indexed_tokens]).type('torch.LongTensor')
            tokens_tensor = tokens_tensor.squeeze()
            assert len(tokens_tensor) == max_num_tokens
            segments_tensors = torch.ones(max_num_tokens)
            segments_tensors = segments_tensors.type('torch.LongTensor')
            input_token_mask = torch.from_numpy(input_token_mask)
            input_token_mask = input_token_mask.type('torch.LongTensor')
            object_input_all[i,:] = tokens_tensor
            segments_tensors_all[i,:] = segments_tensors
            input_token_mask_all[i,:] = input_token_mask
        return object_input_all, segments_tensors_all, input_token_mask_all, input_mask, n_object, object_target, relation_target

    def _LoadImage(self, impath):
        """
        returns: original image, croped and normalized image for model input
        """
        img = Image.open(impath).convert('RGB')
        img_original = self.image_resize_and_crop(img)
        img = self.image_normalize(img_original)
        return img_original, img

    def __getitem__(self, index):
        """
        returns: combination of visual and language input
        """
        datum = self.data[index]
        # caption = os.path.join(self.caption_base_path, datum['caption'])
        imgpath = os.path.join(self.image_base_path, datum['image'])
        img_original, img = self._LoadImage(imgpath)
        object_name_list = datum['object_description']  # object_name, object_description
        object_class_list = datum['object_class']
        relation_list = datum['relation']
        object_input_all, segments_tensors_all, input_token_mask_all, input_mask, n_object, object_target, relation_target = self._LoadObject(object_name_list, object_class_list, relation_list)
        return img_original, img, object_input_all, segments_tensors_all, input_token_mask_all, input_mask, n_object, object_target, relation_target

    def __len__(self):
        return len(self.data)

class LoadCoCoDataset(Dataset):
    def __init__(self, dataset_json_file, image_conf=None):
        """
        Dataset that manages a set of paired images and captions in the COCO dataset

        :param dataset_json_file
        :param image_conf: Dictionary containing 'crop_size' and 'center_crop'
        """
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.image_base_path = data_json['image_base_path']
        self.caption_base_path = data_json['caption_base_path']

        if not image_conf:
            self.image_conf = {}
        else:
            self.image_conf = image_conf

        center_crop = self.image_conf.get('center_crop', False)
        crop_size = self.image_conf.get('crop_size', 224)

        if center_crop:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        else:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])

        RGB_mean = self.image_conf.get('RGB_mean', [0.485, 0.456, 0.406])
        RGB_std = self.image_conf.get('RGB_std', [0.229, 0.224, 0.225])
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

        # tokenizer for Bert model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def _LoadCaption(self, caption):
        """
        returns:
            tokens_tensor: (sequence_max,) tokenized caption
            segments_tensors: (sequence_max,) all 1s
            input_mask: (sequence_max,) 1 for valid tokens; 0 for padding tokens
            nw: int; number of valid tokens in the input caption
        """
        with open(caption) as f:
            lines = f.read().splitlines()
        # randomly choose 1 out of 5 captions in each iteration
        caption = '[CLS] ' + lines[np.random.randint(0, len(lines))] + ' [SEP]'
        # tokenize the sequence
        tokenized_text = self.tokenizer.tokenize(caption)
        nw = len(tokenized_text)
        sequence_max = 32
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        input_mask = np.ones(len(tokenized_text))
        # cut sequence if too long
        p = sequence_max - nw
        if p > 0:
            indexed_tokens = np.pad(indexed_tokens, (0, p), 'constant', constant_values=(0, 0))
            input_mask = np.pad(input_mask, (0, p), 'constant', constant_values=(0, 0))
        elif p < 0:
            indexed_tokens = indexed_tokens[0:p]
            input_mask = input_mask[0:p]
            nw = sequence_max
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.squeeze()
        segments_tensors = torch.ones(sequence_max)
        segments_tensors = segments_tensors.type('torch.LongTensor')
        input_mask = torch.from_numpy(input_mask)
        input_mask = input_mask.type('torch.LongTensor')
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        assert len(tokens_tensor) == sequence_max
        assert len(input_mask) == sequence_max
        assert len(segments_tensors) == sequence_max

        return tokens_tensor, segments_tensors, input_mask, nw

    def _LoadImage(self, impath):
        """
        returns: original image, croped and normalized image for model input
        """
        img = Image.open(impath).convert('RGB')
        img_original = self.image_resize_and_crop(img)
        img = self.image_normalize(img_original)
        return img_original, img

    def __getitem__(self, index):
        """
        returns: combination of visual and language input
        """
        datum = self.data[index]
        caption = os.path.join(self.caption_base_path, datum['caption'])
        imgpath = os.path.join(self.image_base_path, datum['image'])
        caption, token_type, input_mask, nwords = self._LoadCaption(caption) # revise here
        img_original, img = self._LoadImage(imgpath)
        return img_original, img, caption, token_type, input_mask, nwords

    def __len__(self):
        return len(self.data)
