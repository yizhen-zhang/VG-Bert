# Visually Grounded Bert Language Model

This repository is the official implementation of [Explainable Semantic Space by Grounding Language to Vision with Cross-Modal Contrastive Learning](https://openreview.net/forum?id=ljOg2HIBDGH).

To cite this work:
Zhang, Y., Choi, M., Han, K., & Liu, Z. Explainable Semantic Space by Grounding Language toVision with Cross-Modal Contrastive Learning. (accepted by Neurips 2021).

### Abstract
In natural language processing, most models try to learn semantic representa- tions merely from texts. The learned representations encode the “distributional semantics” but fail to connect to any knowledge about the physical world. In contrast, humans learn language by grounding concepts in perception and action and the brain encodes “grounded semantics” for cognition. Inspired by this notion and recent work in vision-language learning, we design a two-stream model for grounding language learning in vision. The model includes a VGG-based visual stream and a Bert-based language stream. The two streams merge into a joint representational space. Through cross-modal contrastive learning, the model first learns to align visual and language representations with the MS COCO dataset. The model further learns to retrieve visual objects with language queries through a cross-modal attention module and to infer the visual relations between the retrieved objects through a bilinear operator with the Visual Genome dataset. After training, the model’s language stream is a stand-alone language model capable of embedding concepts in a visually grounded semantic space. This semantic space manifests principal dimensions explainable with human intuition and neurobiological knowl- edge. Word embeddings in this semantic space are predictive of human-defined norms of semantic features and are segregated into perceptually distinctive clusters. Furthermore, the visually grounded language model also enables compositional language understanding based on visual knowledge and multimodal image search with queries based on images, texts, or their combinations.

## Requirements

The model was trained with Python version: 3.7.4.

`numpy==1.17.2, scipy==1.3.1, torch==1.7.1, torchvision==0.8.2, transformers==4.2.1, Pillow==6.2.0, tokenizers==0.9.4`

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

![visual grounding of natural language](figures/model_structure_horizontal.jpg?raw=true)

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

![visual grounding of natural language](figures/finetune_structure.jpg?raw=true)

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
We include the jupyter notebook scripts for running evaluation tasks in our paper. See README in `evaluation/`.
