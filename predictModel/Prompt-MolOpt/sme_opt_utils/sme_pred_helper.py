import os 
import sys 
import torch as th 
import pandas as pd 
from torch .utils .data import DataLoader 
from rdkit import Chem 
import dgl 
try :
    from tqdm .auto import tqdm 
except ImportError :
    def tqdm (iterable =None ,**kwargs ):
        return iterable if iterable is not None else []

        # Make sure. Run
PROJECT_ROOT =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))
if PROJECT_ROOT not in sys .path :
    sys .path .insert (0 ,PROJECT_ROOT )

from sme_opt_utils .maskgnn import EarlyStopping ,Meter 
from sme_opt_utils .build_data import MolGraphDataset ,collate_molgraphs 

# RGCN Modeland SME_pred_for_mols count, From opt_mol_all_pred Rip Out, AvoidDocumentation.


def SME_pred_for_mols (
smis ,
model_name ,
rgcn_hidden_feats ,
ffn_hidden_feats ,
batch_size =128 ,
lr =0.0003 ,
classification =False ,
device ="cpu",
num_workers =0 ,
num_seeds =10 ,
):
    if num_seeds <=0 :
        raise ValueError ("num_seeds availableCount")
    if device =="auto":
        device ="cuda"if th .cuda .is_available ()else "cpu"
    if device =="cuda":
        if not th .cuda .is_available ():
            print ("Detected PyTorch Use it. CUDA，Switch automatically to CPU。")
            device ="cpu"
        else :
            try :
            # Test DGL Yes CUDA
                _ =dgl .graph (([0 ],[0 ])).to ("cuda")
            except Exception :
                print ("Detected DGL Use it. CUDA，Switch automatically to CPU。")
                device ="cpu"

    args ={
    "device":device ,
    "node_data_field":"node",
    "edge_data_field":"edge",
    "substructure_mask":"smask",
    "num_epochs":500 ,
    "patience":30 ,
    "batch_size":batch_size ,
    "mode":"higher",
    "in_feats":40 ,
    "classification":classification ,
    "rgcn_hidden_feats":rgcn_hidden_feats ,
    "ffn_hidden_feats":ffn_hidden_feats ,
    "rgcn_drop_out":0 ,
    "ffn_drop_out":0 ,
    "lr":lr ,
    "loop":True ,
    "task_name":model_name ,
    }

    # Delay Import, Avoid circulatory dependence
    from sme_opt_utils .maskgnn import RGCN 

    valid_smis =[]
    valid_indices =[]
    for idx ,smi in enumerate (tqdm (smis ,desc ="Could not close temporary folder: %s SMILES",leave =False ,unit ="Article")):
        if Chem .MolFromSmiles (smi ):
            valid_smis .append (smi )
            valid_indices .append (idx )
    if not valid_smis :
        raise ValueError ("InputMediumNo SMILES。")

    y_pred_sum =None 
    dataset =MolGraphDataset (valid_smis )
    data_loader =DataLoader (
    dataset ,
    batch_size =args ["batch_size"],
    collate_fn =collate_molgraphs ,
    num_workers =num_workers ,
    )

    ckpt_dir =os .path .join (PROJECT_ROOT ,"checkpoints","sme")

    for seed in tqdm (range (num_seeds ),desc =f"{model_name} Seed reasoning",unit ="seed"):
        model =RGCN (
        ffn_hidden_feats =args ["ffn_hidden_feats"],
        ffn_dropout =args ["ffn_drop_out"],
        rgcn_node_feats =args ["in_feats"],
        rgcn_hidden_feats =args ["rgcn_hidden_feats"],
        rgcn_drop_out =args ["rgcn_drop_out"],
        classification =args ["classification"],
        )
        stopper =EarlyStopping (
        patience =args ["patience"],
        task_name =args ["task_name"]+"_"+str (seed +1 ),
        mode =args ["mode"],
        filename =os .path .join (ckpt_dir ,f"{model_name}_{seed + 1}_early_stop.pth"),
        )
        model .to (args ["device"])
        stopper .load_checkpoint (model )
        model .eval ()
        eval_meter =Meter ()

        batch_iter =tqdm (
        data_loader ,
        total =len (data_loader ),
        desc =f"Seed {seed + 1} Batch",
        unit ="batch",
        leave =False ,
        )
        for batch_mol_bg in batch_iter :
            batch_mol_bg =batch_mol_bg .to (args ["device"])
            with th .no_grad ():
                rgcn_node_feats =batch_mol_bg .ndata .pop (args ["node_data_field"]).float ().to (args ["device"])
                rgcn_edge_feats =batch_mol_bg .edata .pop (args ["edge_data_field"]).long ().to (args ["device"])
                smask_feats =batch_mol_bg .ndata .pop (args ["substructure_mask"]).unsqueeze (dim =1 ).float ().to (
                args ["device"]
                )

                preds ,weight =model (batch_mol_bg ,rgcn_node_feats ,rgcn_edge_feats ,smask_feats )
                eval_meter .update (preds ,preds )
                th .cuda .empty_cache ()

        _ ,y_pred =eval_meter .compute_metric ("return_pred_true")
        y_pred =th .sigmoid (y_pred ).squeeze ().numpy ()if args ["classification"]else y_pred .squeeze ().numpy ()
        y_pred_sum =y_pred if y_pred_sum is None else y_pred_sum +y_pred 

    y_pred_mean =y_pred_sum /num_seeds 

    # WillpredictionOutcome, Illegal SMILES Use it. NaN Places
    full_preds =[float ("nan")]*len (smis )
    for idx ,pred in zip (valid_indices ,y_pred_mean .tolist ()):
        full_preds [idx ]=pred 
    return full_preds 
