# Visually Grounded Bert Model

This repository is the official implementation of [Explainable Semantic Space by Grounding Language to Vision with Cross-Modal Contrastive Learning]

## Requirements

Python version: 3.7.4. 
`numpy==1.17.2`
`scipy==1.3.1`
`torch==1.7.1`
`torchvision==0.8.2`
`transformers==4.2.1`
`Pillow==6.2.0`
`tokenizers==0.9.4`

## Training

To train the model(s) in the paper, run the following commands:

### stage-1: visual stream pretraining:
```
python visual_stream_pretraining.py \
-a vgg16_attention \
--pretrained \
--batch-size 200 \
--use-position learn \
--lr 0.01 \
/directory-to-ImageNet-dataset/ImageNet2012/
```

### stage-2: two-stream grounding on MS COCO dataset with corss-modal contrastive loss:
```
python run.py \
--stage two_stream_pretraining \
--data-train /directory-to-COCO-dataset/COCO_train2017.json \
--data-val /directory-to-COCO-dataset/COCO_val2017.json \
--optim adam \
--learning-rate 5e-5 \
--batch-size 180 \
--n_epochs 100 \
--pretrained-vgg \
--image-model VGG16_Attention \
--use-position learn \
--language-model Bert_base \
--embedding-dim 768 \
--sigma 0.1 \
--dropout-rate 0.3 \
--base-model-learnable-layers 8 \
--load-pretrained /directory-to-pretrain-image-model/ \
--exp-dir /output-directory/two-stream-pretraining/
```

### stage-3: visual relational grounding on Visual Genome dataset:
```
python run.py \
--stage relational_grounding \
--data-train /directory-to-Visual-Genome-dataset/VG_train_dataset_finalized.json \
--data-val /directory-to-Visual-Genome-dataset/VG_val_dataset_finalized.json \
--optim adam \
--learning-rate 1e-5 \
--batch-size 180 \
--n_epochs 150 \
--pretrained-vgg \
--image-model VGG16_Attention \
--use-position learn \
--language-model Bert_object \
--num-heads 8 \
--embedding-dim 768 \
--subspace-dim 32 \
--relation-num 115 \
--temperature 1 \
--dropout-rate 0.1 \
--base-model-learnable-layers 2 \
--load-pretrained /directory-to-pretrain-two-stream-model/ \
--exp-dir /output-directory/relational-grounding/
```
### Transfer learning for cross-modal image search:
```
python transfer_cross_modal_retrieval.py \
--data-train /directory-to-COCO-dataset/COCO_train2017.json \
--data-val /directory-to-COCO-dataset/COCO_val2017.json \
--optim adam \
--learning-rate 5e-5 \
--batch-size 300 \
--n_epochs 150 \
--pretrained-vgg \
--image-model VGG16_Attention \
--use-position absolute_learn \
--language-model Bert_object \
--num-heads 8 \
--embedding-dim 768 \
--subspace-dim 32 \
--relation-num 115 \
--sigma 0.1 \
--dropout-rate 0.1 \
--load-pretrained /directory-to-pretrain-two-stream-model/ \
--exp-dir /output-directory/cross-modal-retrieval/
```

## Evaluation
See two jupyter notebooks in `evaluation/script`.
