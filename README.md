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
repo_dir = snapshot_download(repo_id="glacier-hf/Glacier-100k-Mi")
sys.path.append(repo_dir)
from data.dataloader import SmilesMoleculeDataset, build_dataloader
from glacier_student import Glacier

model = Glacier.from_pretrained("glacier-hf/Glacier-100k-Mi")

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



Please feel free to download and use these models for your own research purposes. We only ask that you cite our work appropriately if you use it in your work. Thank you for your interest in our research!
