# GLACIER: A Multimodal Student-Teacher Foundation Model for Molecular Property Prediction


<p align="center"><strong>KDD 2026</strong></p>

  <p align="center">
  <a href="https://github.com/eemokey/glacier/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://arxiv.org/abs/2606.11382"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://huggingface.co/glacier-hf/GLACIER-100k-MiniMol'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-F8D44E' alt='HF Model'></a>
  <a href='https://huggingface.co/datasets/glacier-hf/glacier_pretrain_EnamineREAL_100k'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HF-Dataset-F8D44E.svg' alt='HF Dataset'></a>
</p> 

<p align="center"><img src="https://github.com/eemokey/glacier/blob/main/imgs/icon.png" width="400" /></p>

GLACIER (Graph-Language Alignment for Chemical Inference and Exploration using Representations) is a multimodal student-teacher foundation model designed for molecular property prediction. It integrates molecular graphs, SMILES strings, and physicochemical descriptors to learn rich molecular embeddings.



# Environment Setup

### Conda  
```bash
conda env_create -f environment.yml
conda activate glacier
```


### Pip 

```bash
pip install -r requirements.txt
```


#  A basic example of a GLACIER run

```python
import torch 
from huggingface_hub import snapshot_download
import sys

# Download the repository to access custom model code
repo_dir = snapshot_download(repo_id="glacier-hf/GLACIER-100k-MiniMol")
sys.path.append(repo_dir)

from data.dataloader import SmilesMoleculeDataset, build_dataloader
from glacier_student import Glacier

# Load the pretrained GLACIER model
model = Glacier.from_pretrained("glacier-hf/GLACIER-100k-MiniMol")

# Prepare input data
dataset = SmilesMoleculeDataset(smiles=["Cn1c(=O)c2c(ncn2C)n(C)c1=O"])
dataloader = build_dataloader(dataset, batch_size=1)

model.eval()
batch = next(iter(dataloader))
with torch.no_grad():
    embedding = model(batch)
print(embedding)
```

# Citation

Please feel free to download and use these models for your own research purposes. We only ask that you cite our work appropriately if you use it in your work. Thank you for your interest in our research!

```bibtex
@inproceedings{nguyen2026glacier,
  title={GLACIER: A Multimodal Student-Teacher Foundation Model for Molecular Property Prediction},
  author={Emily Nguyen and Yongchan Hong and Harsh Toshniwal and Yan Liu and Andreas Luttens},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’26)},
  year={2026},
  publisher={ACM},
  doi={10.1145/3770855.3819032}
}
```

# License
MIT

