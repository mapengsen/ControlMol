# Controllable Molecular Optimization through Disentangled Visual Representations

---

**ControlMol** is a disentangled molecular optimization repository built around molecular image representations. The current mainline connects the following modules:

- `CGIP` pretrained encoder: maps molecular images into 512-dimensional representations.
- `latent diffusion`: generates candidate molecular images from optimized representations.
- `train_optimization_MMP`: learns representation edits that change only the target property while keeping the other properties as stable as possible.
- `MolScribe`: converts generated images back into SMILES.
- `Prompt-MolOpt`: predicts ADMET properties for candidate molecules for final selection.

This document focuses on the workflow that already exists in the repository and is aligned with the ADMET four-task setting:
`BBBP`, `ESOL`, `hERG`, and `lipop`. The codebase still contains partial support for `Mutagenicity`, but the main commands in this README focus on the four properties above.

## 🪄 Adapt to Your Own Task
This repository is most suitable for the following use cases:

| Use Case | Relevant Module | Description |
| --- | --- | --- |
| Single-target property optimization | `train_optimization_MMP/train.py` + `inference_with_generation.py` | For example, optimize `BBBP` while keeping `ESOL/hERG/lipop` as unchanged as possible |
| Batch candidate generation | `inference_with_generation.py` | Generate candidate molecular images for multiple starting molecules in the test set in one run |
| Image-to-structure parsing | `batch_analyze_molecular_weight_parallel.py` | Use MolScribe to convert generated molecular images into SMILES in batch |
| Final candidate selection | `select_best_decoupled_smiles_prompt.py` | Combine property prediction and similarity constraints to select the best disentangled molecule |

## Installation

### Conda Environment Setup

```bash
conda create -n MDT2 python=3.10
pip install --ignore-installed PyYAML
pip install pip==20.3.3
pip install pytorch_lightning==1.6.1
pip install einops
pip install torchinfo
pip install realesrgan
pip install inflection
pip install blendmodes
pip install lark
pip install opencv-python
pip install opencv-python-headless
pip install torch_fidelity
pip install omegaconf
pip install wandb
pip install tensorboard==2.11.2
pip install pandas
pip install matplotlib
pip install hydra
pip install hydra-core

pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

cd src/taming-transformers
pip install -e .
cd ../..

pip install albumentations==1.1.0
pip install SmilesPE
pip install OpenNMT-py==2.2.0
pip install timm==0.4.12
pip install setuptools==78.1.1
```
```bash
cd ControlMol
```
```bash
cd src/taming-transformers
pip install -e .
cd ../..
```

```bash
export PYTHONPATH="src/taming-transformers:$PYTHONPATH"
```
## Data and ckpt
The data and checkpoints can be found on [Google Drive](https://drive.google.com/drive/folders/1TJh5tNz60hYp3Cp_aMNKzZcZP3MiHcBK?usp=sharing).

## Model Training

## 1️⃣ First Step --> Train the Diffusion Model
If you want to reproduce the full pipeline from scratch, training the diffusion model should be the first step.

### 1) Convert `data/pretrain_imageGeneration.csv` into molecular images first
Before training the diffusion model, batch-render the `smiles` column in the pretraining CSV into PNG files. By default, the images will be written to `data/pretrain_imageGeneration_images/image_file/`.

```bash
python prepare_pretrain_smiles_images.py \
  --input_csv data/pretrain_imageGeneration.csv \
  --output_dir data/pretrain_imageGeneration_images/image_file \
  --smiles_column smiles \
  --name_column index \
  --image_size 256 \
  --chunksize 10000
```

### 2) Train the base latent diffusion model
The following command is the diffusion-model training entry point currently used in this repository. `--data_path` should point to `data/pretrain_imageGeneration_images/image_file`.

```bash
export PYTHONPATH="src/taming-transformers:$PYTHONPATH"

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  main_ldm.py \
  --config config/ldm/mol-ldm-kl-8.yaml \
  --batch_size 32 \
  --epochs 40 \
  --blr 2.5e-7 \
  --weight_decay 0.01 \
  --output_dir log_chekpoints/ \
  --data_path data/pretrain_imageGeneration_images/image_file \
  --eval_freq 1
```

## 2️⃣ Second Step --> Data and Representation Preparation
The goal of this stage is to reorganize the training and test data into property-specific CSV files and prepare the molecular representations and property labels required by later training steps.

### 1) Split the training and test sets by task
Use `split_prompt_molopt_tasks.py` to split `Prompt_MolOpt_train_val_ADMET.csv` and `test_for_ADMET.csv` into per-task files.

Training split:

```bash
python split_prompt_molopt_tasks.py \
  --input data/Prompt_MolOpt_train_val_ADMET.csv \
  --output-dir data/train_valid_data \
  --tasks BBBP ESOL hERG lipop \
  --task-column task_name \
  --filter-column re_group \
  --filter-value origin
```

Test split:

```bash
python split_prompt_molopt_tasks.py \
  --input data/test_for_ADMET.csv \
  --output-dir data/test_for_data \
  --tasks BBBP ESOL hERG lipop \
  --task-column task_name \
  --filter-column re_group \
  --filter-value test_2
```

### 2) Merge them into a unified training file
Use `prepare_all_properties_dataset.py` to merge the four single-task files for `BBBP/ESOL/hERG/lipop` into one master CSV and add the `target_property` column.

```bash
python prepare_all_properties_dataset.py \
  --input-dir data/train_valid_data \
  --properties BBBP ESOL hERG lipop \
  --pattern Prompt_MolOpt_train_val_ADMET_{prop}.csv \
  --output data/train_valid_data/all_train_valid_properties.csv \
  --label-column target_property
```

### 3) Merge the unique SMILES set
Use `merge_admet_smiles.py` to collect `start/final` from the training set and `start` from the test set into a single `smiles` column for later batch representation precomputation.

```bash
python merge_admet_smiles.py \
  --train data/Prompt_MolOpt_train_val_ADMET.csv \
  --test data/test_for_ADMET.csv \
  --output data/merged_admet_smiles.csv
```

### 4) Precompute CGIP representations
Use `train_optimization_MMP/precompute_dynamic_reps_smiles_cache.py` to generate `data/ADMET_smiles_reps.pt`.

```bash
python train_optimization_MMP/precompute_dynamic_reps_smiles_cache.py \
  --csv_path data/merged_admet_smiles.csv \
  --output_path data/ADMET_smiles_reps.pt \
  --encoder_path checkpoints/pretrained_enc_ckpts/CGIP/CGIP.pth \
  --encode_batch_size 128 \
  --image_batch_size 500 \
  --smiles_columns smiles
```

### 5) Train the representation property predictor
Use `train_optimization_MMP/train_property_predictor.py` to train a multi-task property predictor on top of the precomputed representations and produce `best.pt`.

```bash
python train_optimization_MMP/train_property_predictor.py \
  train \
  --csv_path data/ADMET_pred-all.csv \
  --representation_path data/ADMET_smiles_reps.pt \
  --properties BBBP,ESOL,hERG,lipop \
  --save_dir results/rep_property_predictor/admet_4tasks \
  --use_start \
  --start_smiles_column smiles \
  --start_prefix smiles \
  --property_suffix _pred \
  --split_column "" \
  --random_train_frac 0.95 \
  --random_valid_frac 0.05 \
  --batch_size 512 \
  --epochs 20 \
  --device cuda
```

### 6) Use Prompt-MolOpt to predict and fill molecular property columns
Use `predictModel/Prompt-MolOpt/sme_opt_utils/predict_ADMET_pro.py` to fill predicted properties for the `start/final` molecules and generate the training file used by the optimizer.

```bash
python predictModel/Prompt-MolOpt/sme_opt_utils/predict_ADMET_pro.py \
  --input_csv data/train_valid_data/all_train_valid_properties.csv \
  --smiles_columns start,final \
  --output_csv data/train_valid_data/all_train_valid_properties_with_props.csv \
  --hyperparam_pickle predictModel/Prompt-MolOpt/checkpoints/hyperparameter_ADMET_data_for_MGA.pkl \
  --model_names BBBP,ESOL,hERG,lipop \
  --batch_size 256 \
  --device cuda \
  --num_workers 0
```

## 3️⃣ Third Step --> Disentangled Optimization and Selection
The goal of this stage is to learn representation optimization, generate candidate images, convert them back into SMILES, and select the best molecule that satisfies the constraints.

### 1) Train the disentangled representation optimizer

- Input the precomputed representations and the training CSV with property labels.
- Set `property_label_column=target_property`.
- Use `mask_span` to control the editable window size for each property within the 512-dimensional representation.
- Drive disentangled optimization with a target-property loss and invariant-property losses.

The following example uses `BBBP`. For other properties, only replace `--target_property` and the output directory:

```bash
python train_optimization_MMP/train.py \
  --data_path data/train_valid_data/all_train_valid_properties_with_props.csv \
  --target_property BBBP \
  --property_label_column target_property \
  --filter_by_property_label \
  --property_list BBBP,ESOL,hERG,lipop \
  --mask_span 20 \
  --batch_size 32 \
  --num_epochs 100 \
  --lr 1e-4 \
  --encoder_path checkpoints/pretrained_enc_ckpts/CGIP/CGIP.pth \
  --precomputed_reps_path data/ADMET_smiles_reps.pt \
  --property_predictor_checkpoint results/rep_property_predictor/admet_4tasks/best.pt \
  --property_predictor_device cuda \
  --save_dir results/train_optimization_model/BBBP-mask20 \
  --device cuda
```

### 2) Run inference and generation for the target property
Use `train_optimization_MMP/inference_with_generation.py`:

- Input the starting SMILES.
- Encode them into representations with `CGIP`.
- Use the trained representation optimizer to modify the dimensions associated with the target property.
- Feed the edited representations into the diffusion model to generate candidate molecular images in batch.

```bash
python train_optimization_MMP/inference_with_generation.py \
  --optimizer_model_path results/train_optimization_model/BBBP-mask20/final_model.pth \
  --diffusion_config config/ldm/dis_optmization.yaml \
  --diffusion_model_path log_chekpoints/checkpoint-last.pth \
  --encoder_path checkpoints/pretrained_enc_ckpts/CGIP/CGIP.pth \
  --property_predictor_checkpoint results/rep_property_predictor/admet_4tasks/best.pt \
  --target_property BBBP \
  --property_list BBBP,ESOL,hERG,lipop \
  --mask_span 20 \
  --csv_path data/test_for_data/test_for_ADMET_BBBP.csv \
  --csv_path_smiles_column start \
  --num_images 5 \
  --num_original_images 0 \
  --ddim_steps 50 \
  --diffusion_batch_size 1 \
  --max_molecules 100 \
  --output_dir results/ControlMol/BBBP-mask20 \
  --device cuda
```

### 3) Convert candidate images back into SMILES
Use `batch_analyze_molecular_weight_parallel.py` with `MolScribe` to recognize molecular images as SMILES:

```bash
python batch_analyze_molecular_weight_parallel.py \
  --input_output_folder results/ControlMol/BBBP-mask20 \
  --model_path evaluation/MolScribe/ckpt_from_molscribe/swin_base_char_aux_1m680k.pth \
  --output_dir results/ControlMol/BBBP-mask20 \
  --num_processes 4 \
  --batch_size 128
```

### 4) Select the best disentangled molecule
Use `select_best_decoupled_smiles_prompt.py`:

- Automatically fill missing property prediction columns.
- Compare the improvement of the target property.
- Enforce constraints on the remaining properties.
- Filter candidates with fingerprint similarity.
- Output the final CSV of the best molecules.

```bash
python select_best_decoupled_smiles_prompt.py \
  --input_csv results/ControlMol/BBBP-mask20/smiles_predictions_detailed.csv \
  --output_csv results/ControlMol/BBBP-mask20/best_decoupled_smiles.csv \
  --all_success_output_csv results/ControlMol/BBBP-mask20/all_success_decoupled_smiles.csv \
  --failed_output_csv results/ControlMol/BBBP-mask20/failed_decoupled_smiles.csv \
  --target_property BBBP \
  --properties_to_preserve ESOL,hERG,lipop \
  --optimized_smiles_column smiles \
  --source_smiles_column source_smiles \
  --prediction_model_names BBBP,ESOL,hERG,lipop \
  --prediction_hyperparam_pickle predictModel/Prompt-MolOpt/checkpoints/hyperparameter_ADMET_data_for_MGA.pkl \
  --prediction_batch_size 1024 \
  --prediction_num_workers 16 \
  --prediction_device cuda
```
