## Evaluating the visally grounded semantic Space

### PCA analysis

<img src="figures/concrete_abstract_grounded.jpg" width="400">


| Group         | Bert  | Grounded | Relational Grounded |
| ----------    |-------| -------- | --|
| word-level (r)  | 0.1040 | 0.6615 | **0.6948** |
| category-level (r)  | 0.3538 | **0.8749** | 0.8001 |

The human-rated concreteness [[1]](#1) for words in SemCat dataset [[2]](#2), the word category information, and the PCA results for `Bert`, `Grounded`. and `Relational Grounded` models can be found at `data/pca_analysis/`. See usage of these files in `script/pca_analysis.ipynb`.

### Multimodal image search

<img src="figures/cross_modal_search_horizontal.jpg" width="400">

See how to run this expriment in `script/cross_modal_image_search.ipynb`

You can download Open Images Dataset here:
- [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/index.html)

#### Pre-trained Models

You can download pretrained models here:

- [VG relational grounding (after finetuned for multimodal image search)](https://drive.google.com/file/d/1icYBK4MJ7KYWWkoMeJoXFALYBtIn3Yvo/view?usp=sharing)


## References

<a id="1">[1]</a>
Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. Behavior research methods, 46(3), 904-911.

<a id="2">[2]</a>
Şenel, L. K., Utlu, I., Yücesoy, V., Koc, A., & Cukur, T. (2018). Semantic structure and interpretability of word embeddings. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 26(10), 1769-1779.
