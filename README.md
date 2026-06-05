# GLACIER: Graph-Language Alignment for Chemical Inference and Exploration using Representations

# Installation

### Conda Installation 
```bash
conda env_create -f environment.yml
conda activate glacier
```


### Pip Installation

```bash
pip install -r requirements.txt
```


# 🤗 Hugging Face Integration

Our pretrained model weights and processed pretraining datas are publicly available on the HuggingFace:

* **Model Weights:** [glacier-hf/glacier-100k-Mi](https://huggingface.co/glacier-hf/Glacier-100k-Mi) — Pretrained with 100k molecules from Enamine REAL using Minimol as a teacher
* **Datasets:** [glacier-hf/glacier_pretrain_EnamineREAL_100k](https://huggingface.co/datasets/glacier-hf/glacier_pretrain_EnamineREAL_100k) — The curated molecular graphs, SMILES tokens, and physical descriptors used for our KDD evaluation.

Note: You do not need to download these manually; the example scripts will automatically cache them upon execution
