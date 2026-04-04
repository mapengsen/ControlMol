<h3 align="center">Leveraging vision Diffusion Self-conditioned Models for Molecular Design</h3>
<center> <img src="assets/main.png" width="600"> </center>   


---

## News

- 2025-03-04:📂📂 We release the first SelfMol data: [Github](https://github.com/mapengsen/SelfMol)
- 2025-03-04:🚀🚀️We release the first SelfMol base inference code: [Github]([https://arxiv.org/abs/2409.11340](https://github.com/mapengsen/SelfMol))

## What can SelfMol do?
- Antagonist generation
- Analogues Design
- Natural product Design
- Single objective molecule attribute optimization.
  - QED
  - LogP
  - SA
  - MW
  - HBD
  - HBA
  - Molecule Activity
  - Molecule Toxicity
- Muti objective molecule attribute optimization.
  - Dual-target antagonists design


## Requirements Installation

```bash
conda create --name SelfMol python=3.8
conda activate SelfMol
cd SelfMol
pip install -r requirements.txt
```

## Datasets for train and inference

BenchMol provides two new benchmarks, MBANet and StructNet.

#### 1️⃣ The 10M training datasets

| Name   | Link                                                                                         | Description                                                  |
| ------ |----------------------------------------------------------------------------------------------| ------------------------------------------------------------ |
| Pubchem10M | [Google](https://drive.google.com/file/d/1t1Ws-wPYPeeuc8f_SGgnfUCVCzlM_jUJ/view?usp=sharing) | 10 million images of drug-like, bioactive molecules obtained from the PubChem database |

#### 2️⃣ Target datasets
The five target molecule datasets are provided.

| Name  | Link                                                                                         | Description    |
|-------|----------------------------------------------------------------------------------------------|----------------|
| BTK   | [Google](https://drive.google.com/file/d/1t1Ws-wPYPeeuc8f_SGgnfUCVCzlM_jUJ/view?usp=sharing) | The target BTK |
| BACE1 | [Google](https://drive.google.com/file/d/1t1Ws-wPYPeeuc8f_SGgnfUCVCzlM_jUJ/view?usp=sharing) | The target BACE1 |
| HER2  | [Google](https://drive.google.com/file/d/1t1Ws-wPYPeeuc8f_SGgnfUCVCzlM_jUJ/view?usp=sharing) | The target HER2 |
| EGFR  | [Google](https://drive.google.com/file/d/1t1Ws-wPYPeeuc8f_SGgnfUCVCzlM_jUJ/view?usp=sharing) | The target EGFR |
| EP4   | [Google](https://drive.google.com/file/d/1t1Ws-wPYPeeuc8f_SGgnfUCVCzlM_jUJ/view?usp=sharing) | The target EP4 |

#### 3️⃣ Molecular Analogues
The one molecular source analogues are provided at data/.


## How to ues molecules autoencoder model for your model
We train the autoencoder follow [LDM](https://github.com/CompVis/latent-diffusion) training process(config: autoencoder_kl_32x32x4), have trained on 10M molecules.

Some example of encoder source image and reconstruction images:

| Config      | Source                                                                 | Reconstruction                                                          |
|-------------|------------------------------------------------------------------------|--------------------------------------------------------------------------|
| kl_32x32x4  | <img src="assets/testVAE/input/2110.png" width="100">                  | <img src="assets/testVAE/output/2110.png" width="100">                  |
| kl_32x32x4  | <img src="assets/testVAE/input/3960.png" width="100">                  | <img src="assets/testVAE/output/3960.png" width="100">                  |
| kl_32x32x4  | <img src="assets/testVAE/input/11821.png" width="100">                 | <img src="assets/testVAE/output/11821.png" width="100">                 |
| kl_32x32x4  | <img src="assets/testVAE/input/2447.png" width="100">                  | <img src="assets/testVAE/output/2447.png" width="100">                  |


### Download pretrained model
⬇️⬇️ You can download the pre-trained VAE in [Google](https://drive.google.com/file/d/1t1Ws-wPYPeeuc8f_SGgnfUCVCzlM_jUJ/view?usp=sharing).


### Use the VAE
You can use our VAE to your molecule image generation tasks, just follow the steps:
Load the VAE:
```commandline
from latentDiffusion.ldm.util import instantiate_from_config
from omegaconf import OmegaConf
def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return {"model": model}, global_step

config_path = "latentDiffusion/models/first_stage_models/kl-f8/config.yaml"
ckpt_path = "latentDiffusion/logs/imageMol_chembl.ckpt"
config = OmegaConf.load(config_path)
model_info, step = load_model_from_config(config, ckpt_path)
vae = model_info["model"]
```
Use the VAE to encode and decoder molecule images:
```commandline
encoder_x = vae.encode(x).sample().mul_(0.18215)    # (1,3,256,256) --> (1,4,32,32)
encoder_x = encoder_x / 0.18215
decoder_x = self.vae.decode(encoder_x)              # (1,4,32,32)--> (1,3,256,256)
```


## The training of SelfMol
Train SelfMol model following the terminal command:
```bash
export PYTHONPATH="latent_diffusion:$PYTHONPATH"

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0  main_ldm.py \
--config config/ldm/mol-ldm-kl-8.yaml \
--batch_size 64 \
--epochs 400 \
--blr 2.5e-7 --weight_decay 0.01 \
--output_dir log_chekpoints/ \
--data_path /hy-tmp/image_data \
--eval_freq 1
```

⬇️⬇️ You can direct download the pre-trained SelfMol in [Google](https://drive.google.com/file/d/1t1Ws-wPYPeeuc8f_SGgnfUCVCzlM_jUJ/view?usp=sharing).



## The inference of SelfMol
### Target molecule generation
#### BTK target molecules generation:
```bash
python main_ldm_gen.py \
--config config/ldm/condition_generation/mol-ldm-kl-8-gen-BTK.yaml \
--output_dir log_chekpoints/conditionGeneration/generation/BTK \
--evaluate --resume log_chekpoints/checkpoint.pth
```

#### BACE1, HER2, EGFR, EP4 target molecules generation: 
...

### NPs molecule generation



## Molecule optimization





## How to recognition molecule images to smiles?

```bash
python utils/image2smiles.py \
--input_images_path data/test_data \
--image2smiles2image_save_path data/image2smiles2image \
--image2smiles_all data/image2smiles_all.csv \
--image2smiles_validity data/image2smiles_validity.csv \
--image2smiles_unvalidity data/image2smiles_unvalidity.csv
```




# Releases

For more information on BenchMol versions, see the [Releases page](https://github.com/HongxinXiang/BenchMol/blob/master/RELEASE.md).

# Reference

If you find our code or anything else helpful, please do not hesitate to cite the following relevant papers:

```xml

```

# Acknowledge

- [DECIMER](https://github.com/Kohulan/DECIMER-Image_Transformer): Rajan K, et al. "DECIMER.ai - An open platform for automated optical chemical structure identification, segmentation and recognition in scientific publications." Nat. Commun. 14, 5045 (2023).
