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


#  A basic example of a GLACIER run

```Python
import torch 
from huggingface_hub import snapshot_download
import sys
repo_dir = snapshot_download(repo_id="glacier-hf/GLACIER-100k-MiniMol")
sys.path.append(repo_dir)
from data.dataloader import SmilesMoleculeDataset, build_dataloader
from glacier_student import Glacier

model = Glacier.from_pretrained("glacier-hf/GLACIER-100k-MiniMol")

dataset = SmilesMoleculeDataset(smiles=["Cn1c(=O)c2c(ncn2C)n(C)c1=O"])
dataloader = build_dataloader(dataset, batch_size=1)

model.eval()
batch = next(iter(dataloader))
with torch.no_grad():
    embedding = model(batch)
embedding
```



# License
MIT


# Citation

Emily Nguyen, Yongchan Hong, Harsh Toshniwal, Yan Liu, and Andreas
Luttens. 2026. GLACIER: A Multimodal Student-Teacher Foundation Model
for Molecular Property Prediction. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’26), August 09–13, 2026, Jeju Island, Republic of Korea. ACM, New York, NY, USA,
17 pages. https://doi.org/10.1145/3770855.3819032


Please feel free to download and use these models for your own research purposes. We only ask that you cite our work appropriately if you use it in your work. Thank you for your interest in our research!
