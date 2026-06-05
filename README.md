<p align="center">
  <img src="https://github.com/eemokey/glacier/blob/main/imgs/icon.png" width="400" />
</p>

<h1 align="center">GLACIER</h1>

<p align="center">
  <strong>Graph-Language Alignment for Chemical Inference and Exploration using Representations</strong>
</p>

<p align="center">
  <a href="https://github.com/eemokey/glacier/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
</p>

---
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
