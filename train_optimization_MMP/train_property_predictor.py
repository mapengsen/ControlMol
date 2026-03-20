import argparse 
import json 
import math 
import os 
import random 
import sys 
from pathlib import Path 
from dataclasses import dataclass 
from typing import Dict ,Iterable ,List ,Optional ,Sequence ,Tuple 

PROJECT_ROOT =Path (__file__ ).resolve ().parent .parent 
if str (PROJECT_ROOT )not in sys .path :
    sys .path .insert (0 ,str (PROJECT_ROOT ))

import numpy as np 
import pandas as pd 
import torch 
import torch .nn as nn 
import torch .nn .functional as F 
from torch .utils .data import DataLoader ,Dataset 
from tqdm import tqdm 

from train_optimization_MMP .dataset import PROPERTY_CONFIG ,sanitize_property_value 


def set_seed (seed :int )->None :
    random .seed (seed )
    np .random .seed (seed )
    torch .manual_seed (seed )
    torch .cuda .manual_seed_all (seed )


@dataclass 
class RegressionNormalizer :
    mean :float 
    std :float 

    def __post_init__ (self )->None :
        self .std =float (self .std if self .std >1e-8 else 1.0 )

    def normalize (self ,tensor :torch .Tensor )->torch .Tensor :
        return (tensor -self .mean )/self .std 

    def denormalize (self ,tensor :torch .Tensor )->torch .Tensor :
        return tensor *self .std +self .mean 

    def state_dict (self )->Dict [str ,float ]:
        return {"mean":float (self .mean ),"std":float (self .std )}

    @classmethod 
    def from_state_dict (cls ,state :Dict [str ,float ])->"RegressionNormalizer":
        return cls (mean =float (state ["mean"]),std =float (state ["std"]))


class RepresentationStore :
    def __init__ (self ,path :str ,required_smiles :Optional [Iterable [str ]]=None ):
        self .path =path 
        self .required_smiles =set (required_smiles )if required_smiles is not None else None 
        self ._store :Dict [str ,torch .Tensor ]={}

    def load (self )->None :
        raw =torch .load (self .path ,map_location ="cpu")
        if not isinstance (raw ,dict ):
            raise TypeError ("CalculateEssaysDocumentationFormat，available dict[str, Tensor]")
        if self .required_smiles is None :
            self ._store ={k :v .detach ().clone ().float ()for k ,v in raw .items ()}
            return 
        selected :Dict [str ,torch .Tensor ]={}
        missing :List [str ]=[]
        for smiles in self .required_smiles :
            tensor =raw .get (smiles )
            if tensor is None :
                missing .append (smiles )
                continue 
            selected [smiles ]=tensor .detach ().clone ().float ()
        self ._store =selected 
        if missing :
            sample =", ".join (list (missing )[:5 ])
            raise KeyError (f"Below SMILES MissingCalculateEssays: {sample} Wait. {len(missing)} One. ")

    def __contains__ (self ,smiles :str )->bool :
        return smiles in self ._store 

    def get (self ,smiles :str )->torch .Tensor :
        return self ._store [smiles ]

    @property 
    def dim (self )->int :
        if not self ._store :
            raise RuntimeError ("LoadEssaysCache")
        first_key =next (iter (self ._store ))
        return int (self ._store [first_key ].numel ())


class MultiTaskPropertyDataset (Dataset ):
    def __init__ (
    self ,
    dataframe :pd .DataFrame ,
    representations :RepresentationStore ,
    properties :Sequence [str ],
    prefixes :Sequence [str ],
    smiles_columns :Dict [str ,str ],
    property_prefixes :Dict [str ,str ],
    property_suffix :str ,
    ):
        self .df =dataframe .reset_index (drop =True )
        self .representations =representations 
        self .properties =list (properties )
        self .prefixes =list (prefixes )
        self .smiles_columns =smiles_columns 
        self .property_prefixes =property_prefixes 
        self .property_suffix =property_suffix 
        self .smiles_cache :Dict [str ,List [str ]]={}
        self .property_values :Dict [str ,Dict [str ,np .ndarray ]]={}
        self .indices :List [Tuple [int ,str ]]=[]
        self ._prepare ()

    def _prepare (self )->None :
        for prefix in self .prefixes :
            smiles_col =self .smiles_columns [prefix ]
            smiles_series =self .df [smiles_col ].fillna ("").astype (str )
            self .smiles_cache [prefix ]=smiles_series .tolist ()
            self .property_values [prefix ]={}
            prop_prefix =self .property_prefixes [prefix ]
            for prop in self .properties :
                col_name =f"{prop_prefix}_{prop}{self. property_suffix}"
                if col_name not in self .df .columns :
                    raise KeyError (f"DataMediumMissingColumns {col_name}")
                values =self .df [col_name ].apply (sanitize_property_value )
                self .property_values [prefix ][prop ]=values .to_numpy (dtype =np .float32 )

        for row_idx in range (len (self .df )):
            for prefix in self .prefixes :
                smiles =self .smiles_cache [prefix ][row_idx ]
                if not smiles or smiles not in self .representations :
                    continue 
                valid =True 
                for prop in self .properties :
                    if not math .isfinite (float (self .property_values [prefix ][prop ][row_idx ])):
                        valid =False 
                        break 
                if valid :
                    self .indices .append ((row_idx ,prefix ))

    def __len__ (self )->int :
        return len (self .indices )

    def __getitem__ (self ,idx :int )->Tuple [torch .Tensor ,Dict [str ,float ]]:
        row_idx ,prefix =self .indices [idx ]
        smiles =self .smiles_cache [prefix ][row_idx ]
        rep =self .representations .get (smiles )
        targets ={prop :float (self .property_values [prefix ][prop ][row_idx ])for prop in self .properties }
        return rep ,targets 


def build_activation (name :str )->nn .Module :
    name =(name or "gelu").lower ()
    if name =="gelu":
        return nn .GELU ()
    if name =="relu":
        return nn .ReLU ()
    raise ValueError (f"ActivateCount {name}")


class SqueezeExcitation (nn .Module ):
    def __init__ (self ,dim :int ,reduction :int =4 ):
        super ().__init__ ()
        hidden =max (dim //reduction ,1 )
        self .fc1 =nn .Linear (dim ,hidden )
        self .fc2 =nn .Linear (hidden ,dim )

    def forward (self ,x :torch .Tensor )->torch .Tensor :
    # x: [B, D], Every one. Attention.
        scale =self .fc1 (x )
        scale =F .relu (scale )
        scale =torch .sigmoid (self .fc2 (scale ))
        return x *scale 


class ResidualBlockSimple (nn .Module ):
    """Fragment，ForCompatibility with the old version res_mlp"""

    def __init__ (self ,in_dim :int ,out_dim :int ,dropout :float ,activation :str ="relu"):
        super ().__init__ ()
        self .fc =nn .Linear (in_dim ,out_dim )
        self .act =build_activation (activation )
        self .drop =nn .Dropout (dropout )
        self .use_proj =in_dim !=out_dim 
        self .proj =nn .Linear (in_dim ,out_dim )if self .use_proj else nn .Identity ()

    def forward (self ,x :torch .Tensor )->torch .Tensor :
        shortcut =x if not self .use_proj else self .proj (x )
        out =self .fc (x )
        out =self .act (out )
        out =self .drop (out )
        return out +shortcut 


class ResidualBlockDeep (nn .Module ):
    """Layer + Impairment + LayerNorm"""

    def __init__ (self ,dim :int ,dropout :float ,activation :str ="gelu"):
        super ().__init__ ()
        self .fc1 =nn .Linear (dim ,dim )
        self .fc2 =nn .Linear (dim ,dim )
        self .act =build_activation (activation )
        self .drop =nn .Dropout (dropout )
        self .norm =nn .LayerNorm (dim )

    def forward (self ,x :torch .Tensor )->torch .Tensor :
        out =self .fc1 (x )
        out =self .act (out )
        out =self .drop (out )
        out =self .fc2 (out )
        out =self .drop (out )
        out =out +x 
        out =self .norm (out )
        return out 


class DeepResMLPBackbone (nn .Module ):
    """Input → Activate+LayerNorm+Dropout → MultipleFragment"""

    def __init__ (
    self ,
    input_dim :int ,
    hidden_dim :int ,
    num_blocks :int ,
    dropout :float ,
    activation :str ="gelu",
    use_se :bool =False ,
    ):
        super ().__init__ ()
        self .use_se =use_se 
        self .se =SqueezeExcitation (input_dim )if use_se else None 
        self .proj =nn .Linear (input_dim ,hidden_dim )
        self .act =build_activation (activation )
        self .norm =nn .LayerNorm (hidden_dim )
        self .drop =nn .Dropout (dropout )
        self .blocks =nn .ModuleList (
        [ResidualBlockDeep (hidden_dim ,dropout =dropout ,activation =activation )for _ in range (num_blocks )]
        )

    def forward (self ,x :torch .Tensor )->torch .Tensor :
        if self .use_se and self .se is not None :
            x =self .se (x )
        x =self .proj (x )
        x =self .act (x )
        x =self .norm (x )
        x =self .drop (x )
        for block in self .blocks :
            x =block (x )
        return x 


class MMoEBackbone (nn .Module ):
    """Multi-gate Mixture-of-Experts"""

    def __init__ (
    self ,
    input_dim :int ,
    hidden_dim :int ,
    num_experts :int ,
    num_layers :int ,
    task_names :Sequence [str ],
    dropout :float ,
    activation :str ="gelu",
    use_se :bool =False ,
    ):
        super ().__init__ ()
        self .use_se =use_se 
        self .se =SqueezeExcitation (input_dim )if use_se else None 
        self .proj =nn .Linear (input_dim ,hidden_dim )
        self .act =build_activation (activation )
        self .norm =nn .LayerNorm (hidden_dim )
        self .drop =nn .Dropout (dropout )
        self .experts =nn .ModuleList (
        [
        nn .Sequential (
        *[
        ResidualBlockDeep (hidden_dim ,dropout =dropout ,activation =activation )
        for _ in range (num_layers )
        ]
        )
        for _ in range (num_experts )
        ]
        )
        self .gates =nn .ModuleDict ({task :nn .Linear (hidden_dim ,num_experts )for task in task_names })

    def forward (self ,x :torch .Tensor )->Dict [str ,torch .Tensor ]:
        if self .use_se and self .se is not None :
            x =self .se (x )
        base =self .proj (x )
        base =self .act (base )
        base =self .norm (base )
        base =self .drop (base )

        expert_outputs =torch .stack ([expert (base )for expert in self .experts ],dim =1 )# [B, E, D]

        task_features :Dict [str ,torch .Tensor ]={}
        for task ,gate in self .gates .items ():
            weight =torch .softmax (gate (base ),dim =-1 ).unsqueeze (-1 )# [B, E, 1]
            mixed =torch .sum (expert_outputs *weight ,dim =1 )# [B, D]
            task_features [task ]=mixed 
        return task_features 


class MultiTaskMLP (nn .Module ):
    def __init__ (
    self ,
    input_dim :int ,
    hidden_dims :Sequence [int ],
    task_configs :Dict [str ,Dict [str ,str ]],
    dropout :float =0.1 ,
    backbone_type :str ="mlp",
    activation :str ="gelu",
    backbone_width :int =1024 ,
    backbone_layers :int =4 ,
    mmoe_experts :int =4 ,
    mmoe_expert_layers :int =2 ,
    use_se :bool =False ,
    ):
        super ().__init__ ()
        self .backbone_type =backbone_type 
        self .use_se =use_se 

        task_names =list (task_configs .keys ())
        feature_dim :int 

        if backbone_type =="mmoe":
            self .backbone =MMoEBackbone (
            input_dim =input_dim ,
            hidden_dim =backbone_width ,
            num_experts =mmoe_experts ,
            num_layers =mmoe_expert_layers ,
            task_names =task_names ,
            dropout =dropout ,
            activation =activation ,
            use_se =use_se ,
            )
            feature_dim =backbone_width 
        elif backbone_type =="resmlp_deep":
            self .backbone =DeepResMLPBackbone (
            input_dim =input_dim ,
            hidden_dim =backbone_width ,
            num_blocks =backbone_layers ,
            dropout =dropout ,
            activation =activation ,
            use_se =use_se ,
            )
            feature_dim =backbone_width 
        elif backbone_type =="res_mlp":
            layers :List [nn .Module ]=[]
            prev_dim =input_dim 
            for hidden in hidden_dims :
                layers .append (ResidualBlockSimple (prev_dim ,hidden ,dropout ,activation =activation ))
                prev_dim =hidden 
            self .backbone =nn .Sequential (*layers )if layers else nn .Identity ()
            feature_dim =prev_dim 
        else :# mlp
            layers :List [nn .Module ]=[]
            prev_dim =input_dim 
            for hidden in hidden_dims :
                layers .append (nn .Linear (prev_dim ,hidden ))
                layers .append (build_activation (activation ))
                layers .append (nn .Dropout (dropout ))
                prev_dim =hidden 
            self .backbone =nn .Sequential (*layers )if layers else nn .Identity ()
            feature_dim =prev_dim 

        self .heads =nn .ModuleDict ()
        for prop ,cfg in task_configs .items ():
            out_dim =int (cfg .get ("out_dim",1 ))
            self .heads [prop ]=nn .Linear (feature_dim ,out_dim )

        self ._init_weights ()

    def _init_weights (self )->None :
        for module in self .modules ():
            if isinstance (module ,nn .Linear ):
                nn .init .xavier_uniform_ (module .weight )
                if module .bias is not None :
                    nn .init .zeros_ (module .bias )

    def forward (self ,x :torch .Tensor )->Dict [str ,torch .Tensor ]:
        if self .backbone_type =="mmoe":
            features_dict =self .backbone (x )
            return {prop :self .heads [prop ](features_dict [prop ]).squeeze (-1 )for prop in self .heads }
        features =self .backbone (x )
        return {prop :head (features ).squeeze (-1 )for prop ,head in self .heads .items ()}


def build_task_configs (properties :Sequence [str ])->Dict [str ,Dict [str ,str ]]:
    configs :Dict [str ,Dict [str ,str ]]={}
    for prop in properties :
        prop_meta =PROPERTY_CONFIG .get (prop ,{})
        prop_type =prop_meta .get ("type","regression")
        configs [prop ]={"type":prop_type ,"out_dim":1 }
    return configs 


def parse_loss_weights (raw :Optional [str ],properties :Sequence [str ])->Dict [str ,float ]:
    weights ={prop :1.0 for prop in properties }
    if not raw :
        return weights 
    pairs =[seg .strip ()for seg in raw .split (", ")if seg .strip ()]
    for pair in pairs :
        if ": "not in pair :
            raise ValueError (f"WeightFormatError: {pair}，available prop:weight")
        name ,value =pair .split (": ",1 )
        name =name .strip ()
        if name not in weights :
            raise KeyError (f"Unknown property name {name}")
        weights [name ]=float (value )
    return weights 


def make_collate_fn (properties :Sequence [str ]):
    def collate (batch :List [Tuple [torch .Tensor ,Dict [str ,float ]]])->Tuple [torch .Tensor ,Dict [str ,torch .Tensor ]]:
        reps =torch .stack ([item [0 ]for item in batch ],dim =0 )
        targets :Dict [str ,List [float ]]={prop :[]for prop in properties }
        for _ ,value_dict in batch :
            for prop in properties :
                targets [prop ].append (value_dict [prop ])
        stacked ={prop :torch .tensor (values ,dtype =torch .float32 )for prop ,values in targets .items ()}
        return reps ,stacked 

    return collate 


def compute_loss (
outputs :Dict [str ,torch .Tensor ],
targets :Dict [str ,torch .Tensor ],
task_configs :Dict [str ,Dict [str ,str ]],
normalizers :Dict [str ,Optional [RegressionNormalizer ]],
loss_weights :Dict [str ,float ],
)->Tuple [torch .Tensor ,Dict [str ,float ]]:
    total_loss =torch .tensor (0.0 ,device =next (iter (outputs .values ())).device )
    loss_breakdown :Dict [str ,float ]={}
    for prop ,logits in outputs .items ():
        cfg =task_configs [prop ]
        target =targets [prop ]
        weight =float (loss_weights [prop ])
        if cfg ["type"]=="classification":
            loss =F .binary_cross_entropy_with_logits (logits ,target )
        else :
            normalizer =normalizers [prop ]
            if normalizer is None :
                raise RuntimeError (f"{prop} Missing")
            normalized_target =normalizer .normalize (target )
            loss =F .mse_loss (logits ,normalized_target )
        total_loss =total_loss +loss *weight 
        loss_breakdown [prop ]=float (loss .detach ().cpu ())
    return total_loss ,loss_breakdown 


def aggregate_regression_metrics (pred :torch .Tensor ,target :torch .Tensor )->Dict [str ,float ]:
    diff =pred -target 
    mse =torch .mean (diff **2 ).item ()
    mae =torch .mean (diff .abs ()).item ()
    rmse =math .sqrt (max (mse ,0.0 ))
    target_mean =torch .mean (target ).item ()
    ss_tot =torch .sum ((target -target_mean )**2 ).item ()
    ss_res =torch .sum (diff **2 ).item ()
    r2 =1.0 -ss_res /ss_tot if ss_tot >1e-8 else float ("nan")
    return {"rmse":rmse ,"mae":mae ,"r2":r2 }


def aggregate_classification_metrics (prob :torch .Tensor ,target :torch .Tensor )->Dict [str ,float ]:
    pred_label =(prob >=0.5 ).float ()
    accuracy =torch .mean ((pred_label ==target ).float ()).item ()
    metrics :Dict [str ,float ]={"accuracy":accuracy }
    try :
        from sklearn .metrics import roc_auc_score 

        auc =roc_auc_score (target .cpu ().numpy (),prob .cpu ().numpy ())
        metrics ["auc"]=float (auc )
    except Exception :
        metrics ["auc"]=float ("nan")
    return metrics 


def summarize_validation_metrics (
valid_stats :Optional [Dict [str ,object ]],
task_configs :Dict [str ,Dict [str ,str ]],
)->Tuple [Optional [float ],Optional [float ]]:
    """SummaryClassesandReturns RMSE Value，Easy to print logs。"""
    if valid_stats is None :
        return None ,None 
    per_property =valid_stats .get ("per_property_metrics")or {}
    cls_acc :List [float ]=[]
    reg_rmse :List [float ]=[]
    for prop ,cfg in task_configs .items ():
        metrics =per_property .get (prop )or {}
        if cfg ["type"]=="classification":
            acc =metrics .get ("accuracy")
            if acc is not None and math .isfinite (float (acc )):
                cls_acc .append (float (acc ))
        else :
            rmse =metrics .get ("rmse")
            if rmse is not None and math .isfinite (float (rmse )):
                reg_rmse .append (float (rmse ))
    avg_acc =sum (cls_acc )/len (cls_acc )if cls_acc else None 
    avg_rmse =sum (reg_rmse )/len (reg_rmse )if reg_rmse else None 
    return avg_acc ,avg_rmse 


def train_one_epoch (
model :MultiTaskMLP ,
loader :DataLoader ,
optimizer :torch .optim .Optimizer ,
device :torch .device ,
task_configs :Dict [str ,Dict [str ,str ]],
normalizers :Dict [str ,Optional [RegressionNormalizer ]],
loss_weights :Dict [str ,float ],
scaler :Optional [torch .cuda .amp .GradScaler ],
grad_clip :Optional [float ],
progress_desc :Optional [str ]=None ,
)->Dict [str ,float ]:
    model .train ()
    total_loss =0.0 
    total_samples =0 
    running_breakdown :Dict [str ,float ]={prop :0.0 for prop in task_configs }
    progress =tqdm (loader ,desc =progress_desc or "train",leave =False )
    for reps ,targets in progress :
        reps =reps .to (device ,non_blocking =True )
        targets_device ={prop :tensor .to (device ,non_blocking =True )for prop ,tensor in targets .items ()}
        optimizer .zero_grad (set_to_none =True )
        if scaler is not None :
            with torch .cuda .amp .autocast ():
                outputs =model (reps )
                loss ,loss_breakdown =compute_loss (outputs ,targets_device ,task_configs ,normalizers ,loss_weights )
            scaler .scale (loss ).backward ()
            if grad_clip is not None and grad_clip >0 :
                scaler .unscale_ (optimizer )
                torch .nn .utils .clip_grad_norm_ (model .parameters (),grad_clip )
            scaler .step (optimizer )
            scaler .update ()
        else :
            outputs =model (reps )
            loss ,loss_breakdown =compute_loss (outputs ,targets_device ,task_configs ,normalizers ,loss_weights )
            loss .backward ()
            if grad_clip is not None and grad_clip >0 :
                torch .nn .utils .clip_grad_norm_ (model .parameters (),grad_clip )
            optimizer .step ()

        batch_size =reps .size (0 )
        total_loss +=float (loss .detach ().cpu ())*batch_size 
        total_samples +=batch_size 
        for prop ,val in loss_breakdown .items ():
            running_breakdown [prop ]+=float (val )*batch_size 
        progress .set_postfix ({"loss":total_loss /max (total_samples ,1 )})

    averaged ={prop :val /max (total_samples ,1 )for prop ,val in running_breakdown .items ()}
    averaged ["loss"]=total_loss /max (total_samples ,1 )
    return averaged 


@torch .no_grad ()
def evaluate (
model :MultiTaskMLP ,
loader :DataLoader ,
device :torch .device ,
task_configs :Dict [str ,Dict [str ,str ]],
normalizers :Dict [str ,Optional [RegressionNormalizer ]],
loss_weights :Dict [str ,float ],
)->Dict [str ,object ]:
    model .eval ()
    total_loss =0.0 
    total_samples =0 
    loss_breakdown :Dict [str ,float ]={prop :0.0 for prop in task_configs }
    buffers :Dict [str ,Dict [str ,List [torch .Tensor ]]]={
    prop :{"pred":[],"target":[]}for prop in task_configs 
    }

    for reps ,targets in loader :
        reps =reps .to (device ,non_blocking =True )
        targets_device ={prop :tensor .to (device ,non_blocking =True )for prop ,tensor in targets .items ()}
        outputs =model (reps )
        loss ,loss_items =compute_loss (outputs ,targets_device ,task_configs ,normalizers ,loss_weights )
        batch_size =reps .size (0 )
        total_loss +=float (loss .detach ().cpu ())*batch_size 
        total_samples +=batch_size 
        for prop ,val in loss_items .items ():
            loss_breakdown [prop ]+=float (val )*batch_size 
        for prop in task_configs :
            buffers [prop ]["pred"].append (outputs [prop ].detach ().cpu ())
            buffers [prop ]["target"].append (targets_device [prop ].detach ().cpu ())

    metrics :Dict [str ,object ]={
    "loss":total_loss /max (total_samples ,1 ),
    "per_property_loss":{prop :val /max (total_samples ,1 )for prop ,val in loss_breakdown .items ()},
    "per_property_metrics":{},
    }

    for prop ,cfg in task_configs .items ():
        pred_tensor =torch .cat (buffers [prop ]["pred"],dim =0 )
        target_tensor =torch .cat (buffers [prop ]["target"],dim =0 )
        if cfg ["type"]=="classification":
            prob =torch .sigmoid (pred_tensor )
            metrics ["per_property_metrics"][prop ]=aggregate_classification_metrics (prob ,target_tensor )
        else :
            normalizer =normalizers [prop ]
            if normalizer is None :
                raise RuntimeError (f"{prop} Missing")
            pred_denorm =normalizer .denormalize (pred_tensor )
            metrics ["per_property_metrics"][prop ]=aggregate_regression_metrics (pred_denorm ,target_tensor )

    return metrics 


def split_dataframe (
df :pd .DataFrame ,
split_column :Optional [str ],
split_values :Optional [str ],
random_train_frac :float =0.95 ,
random_valid_frac :float =0.05 ,
random_test_frac :float =0.0 ,
)->Tuple [pd .DataFrame ,Optional [pd .DataFrame ],Optional [pd .DataFrame ]]:
    if split_column and split_column not in df .columns :
        raise KeyError (f"DataMediumMissingColumns {split_column}")

    if split_column and split_values :
        value_groups ={k .strip ():[]for k in ["train","valid","test"]}
        for segment in split_values .split ("; "):
            segment =segment .strip ()
            if not segment :
                continue 
            if "="not in segment :
                raise ValueError ("CountFormatavailable train=a,b;valid=c")
            name ,raw_values =segment .split ("=",1 )
            name =name .strip ()
            values =[v .strip ()for v in raw_values .split (", ")if v .strip ()]
            if name not in value_groups :
                raise ValueError (f"Unknown Division Name {name} (Support only train/valid/test)")
            value_groups [name ]=values 
        subsets ={}
        for name ,values in value_groups .items ():
            if not values :
                subsets [name ]=None 
                continue 
            mask =df [split_column ].isin (values )
            subsets [name ]=df [mask ].copy ()
        return subsets .get ("train"),subsets .get ("valid"),subsets .get ("test")

        # fallback: random split with custom fractions
    fractions ={
    "train":float (random_train_frac ),
    "valid":float (random_valid_frac ),
    "test":float (random_test_frac ),
    }
    if any (v <0 for v in fractions .values ()):
        raise ValueError ("available，Inspection random_*_frac Count")
    total_frac =sum (fractions .values ())
    if total_frac <=0 :
        raise ValueError ("and 0")
    normalized ={k :v /total_frac for k ,v in fractions .items ()}

    shuffled =df .sample (frac =1.0 ,random_state =42 ).reset_index (drop =True )
    total =len (shuffled )
    if total ==0 :
        raise ValueError ("DataBlank")

    sizes ={k :int (total *frac )for k ,frac in normalized .items ()}
    assigned =sum (sizes .values ())
    remainder =total -assigned 
    for name in ["train","valid","test"]:
        if remainder <=0 :
            break 
        if fractions [name ]>0 or name =="train":
            sizes [name ]+=1 
            remainder -=1 

    start =0 
    subsets ={}
    for name in ["train","valid","test"]:
        end =start +sizes [name ]
        subset =shuffled .iloc [start :end ].copy ()if sizes [name ]>0 else None 
        subsets [name ]=subset 
        start =end 
    return subsets .get ("train"),subsets .get ("valid"),subsets .get ("test")


def compute_regression_normalizers (
train_df :pd .DataFrame ,
properties :Sequence [str ],
prefixes :Sequence [str ],
property_prefixes :Dict [str ,str ],
property_suffix :str ,
)->Dict [str ,Optional [RegressionNormalizer ]]:
    normalizers :Dict [str ,Optional [RegressionNormalizer ]]={}
    for prop in properties :
        cfg =PROPERTY_CONFIG .get (prop ,{})
        if cfg .get ("type","regression")!="regression":
            normalizers [prop ]=None 
            continue 
        values :List [float ]=[]
        for prefix in prefixes :
            col =f"{property_prefixes[prefix]}_{prop}{property_suffix}"
            if col not in train_df .columns :
                continue 
            series =train_df [col ].apply (sanitize_property_value )
            values .extend (series .tolist ())
        if not values :
            raise ValueError (f"{prop} MissingForIt is. .. TrainingData")
        arr =np .asarray (values ,dtype =np .float32 )
        normalizers [prop ]=RegressionNormalizer (mean =float (arr .mean ()),std =float (arr .std ()))
    return normalizers 


def prepare_datasets (
args :argparse .Namespace ,
task_configs :Dict [str ,Dict [str ,str ]],
)->Tuple [
MultiTaskPropertyDataset ,
Optional [MultiTaskPropertyDataset ],
Optional [MultiTaskPropertyDataset ],
Dict [str ,Optional [RegressionNormalizer ]],
RepresentationStore ,
]:
    properties =args .properties 
    prefixes :List [str ]=[]
    smiles_columns :Dict [str ,str ]={}
    property_prefixes :Dict [str ,str ]={}
    property_suffix =args .property_suffix or ""
    if args .use_start :
        prefixes .append ("start")
        smiles_columns ["start"]=args .start_smiles_column 
        property_prefixes ["start"]=args .start_prefix 
    if args .use_final :
        prefixes .append ("final")
        smiles_columns ["final"]=args .final_smiles_column 
        property_prefixes ["final"]=args .final_prefix 
    if not prefixes :
        raise ValueError ("toUse it. start or final MediumIt is. .. ")

    required_columns =set ()
    if args .split_column :
        required_columns .add (args .split_column )
    for prefix in prefixes :
        required_columns .add (smiles_columns [prefix ])
        for prop in properties :
            required_columns .add (f"{property_prefixes[prefix]}_{prop}{property_suffix}")

    df =pd .read_csv (args .csv_path ,usecols =sorted (required_columns ))
    train_df ,valid_df ,test_df =split_dataframe (
    df ,
    split_column =args .split_column ,
    split_values =args .split_values ,
    random_train_frac =args .random_train_frac ,
    random_valid_frac =args .random_valid_frac ,
    random_test_frac =args .random_test_frac ,
    )
    if train_df is None or train_df .empty :
        raise ValueError ("TrainingBlank，InspectionCount")

    normalizers =compute_regression_normalizers (train_df ,properties ,prefixes ,property_prefixes ,property_suffix )

    required_smiles :List [str ]=[]
    for subset in [train_df ,valid_df ,test_df ]:
        if subset is None :
            continue 
        for prefix in prefixes :
            col =smiles_columns [prefix ]
            required_smiles .extend (subset [col ].dropna ().astype (str ).tolist ())
    representation_store =RepresentationStore (args .representation_path ,required_smiles )
    representation_store .load ()

    train_dataset =MultiTaskPropertyDataset (
    dataframe =train_df ,
    representations =representation_store ,
    properties =properties ,
    prefixes =prefixes ,
    smiles_columns =smiles_columns ,
    property_prefixes =property_prefixes ,
    property_suffix =property_suffix ,
    )

    valid_dataset =(
    None 
    if valid_df is None or valid_df .empty 
    else MultiTaskPropertyDataset (
    dataframe =valid_df ,
    representations =representation_store ,
    properties =properties ,
    prefixes =prefixes ,
    smiles_columns =smiles_columns ,
    property_prefixes =property_prefixes ,
    property_suffix =property_suffix ,
    )
    )

    test_dataset =(
    None 
    if test_df is None or test_df .empty 
    else MultiTaskPropertyDataset (
    dataframe =test_df ,
    representations =representation_store ,
    properties =properties ,
    prefixes =prefixes ,
    smiles_columns =smiles_columns ,
    property_prefixes =property_prefixes ,
    property_suffix =property_suffix ,
    )
    )

    return train_dataset ,valid_dataset ,test_dataset ,normalizers ,representation_store 


def save_checkpoint (
path :str ,
model :MultiTaskMLP ,
optimizer :torch .optim .Optimizer ,
epoch :int ,
best_metric :float ,
task_configs :Dict [str ,Dict [str ,str ]],
normalizers :Dict [str ,Optional [RegressionNormalizer ]],
args :argparse .Namespace ,
)->None :
    normalizer_state ={
    prop :norm .state_dict ()if norm is not None else None for prop ,norm in normalizers .items ()
    }
    payload ={
    "epoch":epoch ,
    "best_metric":best_metric ,
    "model_state":model .state_dict (),
    "optimizer_state":optimizer .state_dict (),
    "task_configs":task_configs ,
    "normalizers":normalizer_state ,
    "properties":args .properties ,
    "model_config":{
    "input_dim":args .input_dim ,
    "hidden_dims":args .hidden_dims ,
    "dropout":args .dropout ,
    "backbone_type":args .backbone ,
    "activation":args .activation ,
    "backbone_width":args .backbone_width ,
    "backbone_layers":args .backbone_layers ,
    "mmoe_experts":args .mmoe_experts ,
    "mmoe_expert_layers":args .mmoe_expert_layers ,
    "use_se":args .use_se ,
    },
    "training_args":{
    "loss_weights":args .loss_weights_raw ,
    "use_start":args .use_start ,
    "use_final":args .use_final ,
    "start_prefix":args .start_prefix ,
    "final_prefix":args .final_prefix ,
    "start_smiles_column":args .start_smiles_column ,
    "final_smiles_column":args .final_smiles_column ,
    "property_suffix":args .property_suffix ,
    "split_column":args .split_column ,
    "split_values":args .split_values ,
    "backbone":args .backbone ,
    "activation":args .activation ,
    "backbone_width":args .backbone_width ,
    "backbone_layers":args .backbone_layers ,
    "mmoe_experts":args .mmoe_experts ,
    "mmoe_expert_layers":args .mmoe_expert_layers ,
    "use_se":args .use_se ,
    },
    }
    torch .save (payload ,path )


def train_command (args :argparse .Namespace )->None :
    set_seed (args .seed )
    task_configs =build_task_configs (args .properties )
    (
    train_dataset ,
    valid_dataset ,
    test_dataset ,
    normalizers ,
    representation_store ,
    )=prepare_datasets (args ,task_configs )

    args .input_dim =representation_store .dim 
    model =MultiTaskMLP (
    input_dim =args .input_dim ,
    hidden_dims =args .hidden_dims ,
    task_configs =task_configs ,
    dropout =args .dropout ,
    backbone_type =args .backbone ,
    activation =args .activation ,
    backbone_width =args .backbone_width ,
    backbone_layers =args .backbone_layers ,
    mmoe_experts =args .mmoe_experts ,
    mmoe_expert_layers =args .mmoe_expert_layers ,
    use_se =args .use_se ,
    )

    device =torch .device (args .device if args .device else ("cuda"if torch .cuda .is_available ()else "cpu"))
    model .to (device )

    optimizer =torch .optim .AdamW (model .parameters (),lr =args .lr )
    scheduler =None 
    if args .lr_scheduler =="cosine":
        scheduler =torch .optim .lr_scheduler .CosineAnnealingLR (optimizer ,T_max =args .epochs )
    elif args .lr_scheduler =="step":
        scheduler =torch .optim .lr_scheduler .StepLR (optimizer ,step_size =max (args .epochs //3 ,1 ),gamma =0.2 )

    scaler =torch .cuda .amp .GradScaler ()if args .use_amp and device .type =="cuda"else None 
    loss_weights =parse_loss_weights (args .loss_weights_raw ,args .properties )
    args .loss_weights =loss_weights 

    worker_count =args .num_workers 
    if worker_count is None or worker_count <0 :
        cpu_count =os .cpu_count ()or 1 
        worker_count =max (1 ,cpu_count )
    args .num_workers =worker_count 
    print (f"DataLoad worker Count: {args. num_workers}",flush =True )

    collate_fn =make_collate_fn (args .properties )
    train_loader =DataLoader (
    train_dataset ,
    batch_size =args .batch_size ,
    shuffle =True ,
    num_workers =args .num_workers ,
    pin_memory =(device .type =="cuda"),
    collate_fn =collate_fn ,
    )
    valid_loader =(
    None 
    if valid_dataset is None 
    else DataLoader (
    valid_dataset ,
    batch_size =args .eval_batch_size ,
    shuffle =False ,
    num_workers =args .num_workers ,
    pin_memory =(device .type =="cuda"),
    collate_fn =collate_fn ,
    )
    )
    test_loader =(
    None 
    if test_dataset is None 
    else DataLoader (
    test_dataset ,
    batch_size =args .eval_batch_size ,
    shuffle =False ,
    num_workers =args .num_workers ,
    pin_memory =(device .type =="cuda"),
    collate_fn =collate_fn ,
    )
    )

    os .makedirs (args .save_dir ,exist_ok =True )
    history :List [Dict [str ,object ]]=[]
    best_val =float ("inf")
    best_path =os .path .join (args .save_dir ,"best.pt")
    last_path =os .path .join (args .save_dir ,"last.pt")
    loss_log_path =os .path .join (args .save_dir ,"epoch_loss.txt")
    with open (loss_log_path ,"w",encoding ="utf-8")as f :
        f .write ("epoch\ttrain_loss\tvalid_loss\n")

    for epoch in range (1 ,args .epochs +1 ):
        train_stats =train_one_epoch (
        model ,
        train_loader ,
        optimizer ,
        device ,
        task_configs ,
        normalizers ,
        loss_weights ,
        scaler ,
        args .grad_clip ,
        progress_desc =f"train[{epoch}/{args. epochs}]",
        )
        if scheduler is not None :
            scheduler .step ()
        if valid_loader is not None :
            valid_stats =evaluate (model ,valid_loader ,device ,task_configs ,normalizers ,loss_weights )
        else :
            valid_stats =None 

        avg_acc ,avg_rmse =summarize_validation_metrics (valid_stats ,task_configs )

        try :
            train_loss =float (train_stats .get ("loss"))
        except (TypeError ,ValueError ):
            train_loss =float ("nan")
        summary =[f"Epoch {epoch}/{args. epochs}",f"train_loss={train_loss: .4f}"]
        valid_loss =float ("nan")
        if valid_stats is not None :
            try :
                valid_loss =float (valid_stats .get ("loss"))
            except (TypeError ,ValueError ):
                valid_loss =float ("nan")
            summary .append (f"valid_loss={valid_loss: .4f}")
            if avg_acc is not None :
                summary .append (f"valid_acc_mean={avg_acc: .4f}")
            if avg_rmse is not None :
                summary .append (f"valid_rmse_mean={avg_rmse: .4f}")
        print (" | ".join (summary ),flush =True )
        with open (loss_log_path ,"a",encoding ="utf-8")as f :
            f .write (f"{epoch}\t{train_loss: .6f}\t{valid_loss: .6f}\n")

        record ={"epoch":epoch ,"train":train_stats ,"valid":valid_stats }
        history .append (record )

        with open (os .path .join (args .save_dir ,"training_history.json"),"w",encoding ="utf-8")as f :
            json .dump (history ,f ,ensure_ascii =False ,indent =2 )

        if valid_stats is not None and valid_stats ["loss"]<best_val :
            best_val =float (valid_stats ["loss"])
            save_checkpoint (
            best_path ,
            model ,
            optimizer ,
            epoch ,
            best_val ,
            task_configs ,
            normalizers ,
            args ,
            )

        save_checkpoint (
        last_path ,
        model ,
        optimizer ,
        epoch ,
        best_val ,
        task_configs ,
        normalizers ,
        args ,
        )

    if test_loader is not None :
        best_payload =torch .load (best_path ,map_location =device )
        model .load_state_dict (best_payload ["model_state"])
        test_stats =evaluate (model ,test_loader ,device ,task_configs ,normalizers ,loss_weights )
        with open (os .path .join (args .save_dir ,"test_metrics.json"),"w",encoding ="utf-8")as f :
            json .dump (test_stats ,f ,ensure_ascii =False ,indent =2 )


def load_checkpoint (path :str ,device :torch .device )->Tuple [Dict [str ,object ],MultiTaskMLP ,Dict [str ,Optional [RegressionNormalizer ]]]:
    payload =torch .load (path ,map_location =device )
    properties :List [str ]=payload ["properties"]
    task_configs :Dict [str ,Dict [str ,str ]]=payload ["task_configs"]
    model_cfg =payload ["model_config"]
    training_args =payload .get ("training_args",{})
    hidden_dims_cfg =model_cfg .get ("hidden_dims",[])
    hidden_dims =[int (x )for x in hidden_dims_cfg ]if isinstance (hidden_dims_cfg ,(list ,tuple ))else []
    backbone_type =(
    model_cfg .get ("backbone_type")
    or training_args .get ("backbone")
    or ("res_mlp"if model_cfg .get ("use_residual")else "mlp")
    )
    activation =model_cfg .get ("activation")or training_args .get ("activation")or "relu"
    backbone_width =int (
    model_cfg .get (
    "backbone_width",
    hidden_dims [-1 ]if hidden_dims else model_cfg .get ("input_dim",0 ),
    )
    )
    backbone_layers =int (model_cfg .get ("backbone_layers",training_args .get ("backbone_layers",1 )))
    mmoe_experts =int (model_cfg .get ("mmoe_experts",training_args .get ("mmoe_experts",4 )))
    mmoe_expert_layers =int (model_cfg .get ("mmoe_expert_layers",training_args .get ("mmoe_expert_layers",2 )))
    use_se =bool (model_cfg .get ("use_se",training_args .get ("use_se",False )))
    model =MultiTaskMLP (
    input_dim =int (model_cfg ["input_dim"]),
    hidden_dims =hidden_dims ,
    task_configs =task_configs ,
    dropout =float (model_cfg ["dropout"]),
    backbone_type =backbone_type ,
    activation =activation ,
    backbone_width =backbone_width ,
    backbone_layers =backbone_layers ,
    mmoe_experts =mmoe_experts ,
    mmoe_expert_layers =mmoe_expert_layers ,
    use_se =use_se ,
    )
    model .load_state_dict (payload ["model_state"])
    normalizers :Dict [str ,Optional [RegressionNormalizer ]]={}
    for prop in properties :
        state =payload ["normalizers"].get (prop )
        normalizers [prop ]=RegressionNormalizer .from_state_dict (state )if state is not None else None 
    return payload ,model ,normalizers 


def predict_command (args :argparse .Namespace )->None :
    device =torch .device (args .device if args .device else ("cuda"if torch .cuda .is_available ()else "cpu"))
    payload ,model ,normalizers =load_checkpoint (args .checkpoint ,device )
    properties :List [str ]=payload ["properties"]
    model .to (device )
    model .eval ()

    smiles_list :List [str ]=[]
    if args .input_csv :
        df =pd .read_csv (args .input_csv ,usecols =[args .smiles_column ])
        smiles_list .extend (df [args .smiles_column ].dropna ().astype (str ).tolist ())
    if args .smiles :
        smiles_list .extend (args .smiles )
    if not smiles_list :
        raise ValueError ("Not provided SMILES Input")

    unique_smiles =list (dict .fromkeys (smiles_list ))
    store =RepresentationStore (args .representation_path ,unique_smiles )
    store .load ()

    batch_size =args .batch_size 
    outputs :List [Dict [str ,float ]]=[]
    for start in range (0 ,len (unique_smiles ),batch_size ):
        batch_smiles =unique_smiles [start :start +batch_size ]
        reps =torch .stack ([store .get (smi )for smi in batch_smiles ],dim =0 ).to (device )
        with torch .no_grad ():
            pred_dict =model (reps )
        prob_dict :Dict [str ,torch .Tensor ]={}
        for prop ,logits in pred_dict .items ():
            if payload ["task_configs"][prop ]["type"]=="classification":
                prob_dict [prop ]=torch .sigmoid (logits )
            else :
                prob_dict [prop ]=normalizers [prop ].denormalize (logits .cpu ())
        for idx ,smiles in enumerate (batch_smiles ):
            result ={"smiles":smiles }
            for prop in properties :
                if payload ["task_configs"][prop ]["type"]=="classification":
                    prob =prob_dict [prop ][idx ].detach ().cpu ().item ()
                    result [f"{prop}_prob"]=prob 
                    result [f"{prop}_label"]=1.0 if prob >=0.5 else 0.0 
                else :
                    result [f"{prop}_pred"]=prob_dict [prop ][idx ].detach ().cpu ().item ()
            outputs .append (result )

    result_df =pd .DataFrame (outputs )
    if args .output_path :
        os .makedirs (os .path .dirname (args .output_path )or ".",exist_ok =True )
        result_df .to_csv (args .output_path ,index =False )
    else :
        print (result_df .to_string (index =False ))


def build_arg_parser ()->argparse .ArgumentParser :
    parser =argparse .ArgumentParser (description ="MultitaskProjectionsTrainingand")
    subparsers =parser .add_subparsers (dest ="command",required =True )

    train_parser =subparsers .add_parser ("train",help ="TrainingMultitaskProjections")
    train_parser .add_argument ("--csv_path",type =str ,required =True ,help ="Organisation SMILES CSV")
    train_parser .add_argument ("--representation_path",type =str ,required =True ,help ="CalculateEssays.pt Path")
    train_parser .add_argument (
    "--properties",
    type =lambda s :[p .strip ()for p in s .split (", ")if p .strip ()],
    required =True ,
    help ="Comma SeparatedIt is. .. Columns，For example. .. BBBP, ESOL, hERG, lipop, Mutagenicity",
    )
    train_parser .add_argument ("--save_dir",type =str ,required =True ,help ="SaveModelandIt is. .. ")
    train_parser .add_argument ("--batch_size",type =int ,default =512 )
    train_parser .add_argument ("--eval_batch_size",type =int ,default =1024 )
    train_parser .add_argument ("--epochs",type =int ,default =20 )
    train_parser .add_argument ("--lr",type =float ,default =1e-3 )
    train_parser .add_argument (
    "--hidden_dims",
    type =lambda s :[int (x )for x in s .split (", ")]if isinstance (s ,str )else [],
    default ="512, 256",
    help ="TotalLayer，Comma Separated，For example. .. 512, 256 or 512, 128",
    )
    train_parser .add_argument ("--dropout",type =float ,default =0.2 )
    train_parser .add_argument (
    "--backbone",
    type =str ,
    choices =["mlp","res_mlp","resmlp_deep","mmoe"],
    default ="mlp",
    help ="Selection：Standards MLP、Impairment MLP、Impairment MLP(resmlp_deep) or MMoE",
    )
    train_parser .add_argument (
    "--activation",
    type =str ,
    choices =["relu","gelu"],
    default ="gelu",
    help ="ActivateCount",
    )
    train_parser .add_argument (
    "--backbone_width",
    type =int ,
    default =1024 ,
    help ="Impairment/ExpertsIt is. .. ，For example. .. 1024",
    )
    train_parser .add_argument (
    "--backbone_layers",
    type =int ,
    default =4 ,
    help ="ImpairmentIt is. .. FragmentCount",
    )
    train_parser .add_argument (
    "--mmoe_experts",
    type =int ,
    default =4 ,
    help ="MMoE ExpertsCount",
    )
    train_parser .add_argument (
    "--mmoe_expert_layers",
    type =int ,
    default =2 ,
    help ="Every one. MMoE ExpertsIt is. .. FragmentCount",
    )
    train_parser .add_argument (
    "--use_se",
    action ="store_true",
    default =False ,
    help ="availableYesavailableInputAdd SE Attention. ",
    )
    train_parser .add_argument (
    "--num_workers",
    type =int ,
    default =None ,
    help ="DataLoad worker Count，DefaultUse CPU Core；available 0 Use it. ProcessLoad",
    )
    train_parser .add_argument ("--device",type =str ,default =None ,help ="cuda or cpu，Default")
    train_parser .add_argument ("--split_column",type =str ,default ="group")
    train_parser .add_argument (
    "--split_values",
    type =str ,
    default ="train=train; valid=valid",
    help ="UseColumnsIt is time. Value，Format: train=train1, train2; valid=valid; test=test",
    )
    train_parser .add_argument (
    "--random_train_frac",
    type =float ,
    default =0.95 ,
    help ="ColumnsTrainingIt is. .. ，Default 0.95；Willand valid/test Let us do it together. ",
    )
    train_parser .add_argument (
    "--random_valid_frac",
    type =float ,
    default =0.05 ,
    help ="ColumnsAuthenticationIt is. .. ，Default 0.05；Willand train/test Let us do it together. ",
    )
    train_parser .add_argument (
    "--random_test_frac",
    type =float ,
    default =0.0 ,
    help ="ColumnsTestIt is. .. ，Default 0",
    )
    train_parser .add_argument ("--seed",type =int ,default =42 )
    train_parser .add_argument ("--use_start",action ="store_true",default =False )
    train_parser .add_argument ("--use_final",action ="store_true",default =False )
    train_parser .add_argument ("--start_prefix",type =str ,default ="start")
    train_parser .add_argument ("--final_prefix",type =str ,default ="final")
    train_parser .add_argument ("--start_smiles_column",type =str ,default ="start")
    train_parser .add_argument ("--final_smiles_column",type =str ,default ="final")
    train_parser .add_argument ("--grad_clip",type =float ,default =None )
    train_parser .add_argument ("--use_amp",action ="store_true",default =False )
    train_parser .add_argument (
    "--lr_scheduler",
    type =str ,
    choices =["none","cosine","step"],
    default ="none",
    )
    train_parser .add_argument ("--loss_weights",dest ="loss_weights_raw",type =str ,default =None )
    train_parser .add_argument (
    "--property_suffix",
    type =str ,
    default ="",
    help ="ColumnsIt is. .. ，For example. .. Columnsavailable smiles_BBBP_pred available _pred",
    )

    predict_parser =subparsers .add_parser ("predict",help ="Use it. TrainingIt is. .. ModelCLIProjections")
    predict_parser .add_argument ("--checkpoint",type =str ,required =True )
    predict_parser .add_argument ("--representation_path",type =str ,required =True )
    predict_parser .add_argument ("--input_csv",type =str ,default =None ,help ="Organisation SMILES CSV")
    predict_parser .add_argument ("--smiles_column",type =str ,default ="smiles")
    predict_parser .add_argument ("--smiles",nargs ="*",default =[])
    predict_parser .add_argument ("--batch_size",type =int ,default =512 )
    predict_parser .add_argument ("--output_path",type =str ,default =None )
    predict_parser .add_argument ("--device",type =str ,default =None )

    return parser 


def main ()->None :
    parser =build_arg_parser ()
    args =parser .parse_args ()
    if args .command =="train":
        if not args .use_start and not args .use_final :
            parser .error ("train toUse it. --use_start or --use_final")
        if isinstance (args .hidden_dims ,str ):
            args .hidden_dims =[int (x )for x in args .hidden_dims .split (", ")if x .strip ()]
        train_command (args )
    elif args .command =="predict":
        predict_command (args )
    else :
        raise ValueError (f"Unknown command {args. command}")


if __name__ =="__main__":
    main ()
