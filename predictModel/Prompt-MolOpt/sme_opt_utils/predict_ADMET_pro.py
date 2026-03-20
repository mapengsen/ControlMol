import argparse 
import os 
import sys 
import pickle as pkl 
import pandas as pd 
try :
    from tqdm .auto import tqdm 
except ImportError :
# It is easy. , Not installed tqdm Directly returns the original repeater
    def tqdm (iterable =None ,**kwargs ):
        return iterable if iterable is not None else []

        # Make sure. FromRun,
CURRENT_DIR =os .path .dirname (os .path .abspath (__file__ ))
if CURRENT_DIR not in sys .path :
    sys .path .insert (0 ,CURRENT_DIR )

from sme_pred_helper import SME_pred_for_mols 


def parse_args ():
    parser =argparse .ArgumentParser (description ="Projections ADMET Single Task GNN Count（ProjectionsMultiple）")
    parser .add_argument ("--input_csv",required =True ,help ="Include Moleculars SMILES Input CSV Path")
    parser .add_argument ("--smiles_column",default ="start",help ="Single SMILES Columns（and smiles_columns Two and one. ）")
    parser .add_argument ("--smiles_columns",default =None ,
    help ="Multiple SMILES Columns，Comma Separated，For example. .. start, final；If providedWill smiles_column")
    parser .add_argument ("--output_csv",required =True ,help ="ProjectionsOutcome CSV Path")
    parser .add_argument ("--hyperparam_pickle",default ="result/hyperparameter_BBBP. pkl",
    help ="Count pkl Path，DefaultUse BBBP Documentation")
    parser .add_argument ("--model_name",default ="BBBP",help ="（CompatibilitySingle Task）Model，ForLoad checkpoints/sme/{model_name}_*.pth")
    parser .add_argument ("--model_names",default =None ,
    help ="Comma SeparatedIt is. .. MultipleModel，For example. .. BBBP, ESOL, hERG, lipop；If provided，Will --model_name")
    parser .add_argument ("--batch_size",type =int ,default =256 ,help ="ProjectionsIt is time. batch size")
    parser .add_argument ("--device",default ="cuda",help ="RunEquipment，Default cpu；Like CUDA Version PyTorch/DGL Visible cuda")
    parser .add_argument ("--num_workers",type =int ,default =0 ,help ="DataLoader worker Count，Raise CPU Processing")
    parser .add_argument ("--num_seeds",type =int ,default =10 ,help ="UseOne. checkpoint CLI ensemble，Default 10")
    return parser .parse_args ()


def main ():
    args =parse_args ()

    df =pd .read_csv (args .input_csv )
    if args .smiles_columns :
        smiles_cols =[c .strip ()for c in args .smiles_columns .split (", ")if c .strip ()]
    else :
    # CompatibilityUse it. --smiles_column MediumUse it. Comma SeparatedMultiplecolumns
        if ", "in args .smiles_column :
            smiles_cols =[c .strip ()for c in args .smiles_column .split (", ")if c .strip ()]
        else :
            smiles_cols =[args .smiles_column ]
    for col in smiles_cols :
        if col not in df .columns :
            raise ValueError (f"InputDocumentationMediumFoundColumns: {col}")

    with open (args .hyperparam_pickle ,"rb")as f :
        hyper =pkl .load (f )

        # predictionTaskscolumns
    if args .model_names :
        task_list =[t .strip ()for t in args .model_names .split (", ")if t .strip ()]
    else :
        task_list =[args .model_name ]

        # OutcomeCollection, Inputcolumns（Like target_property Wait. columns）, AvoidTrainingcolumns
    out_df =df .copy ()

    for task in tqdm (task_list ,desc ="ProjectionsTasks",unit ="task"):
    # CompatibilityDocumentation: classification Tag, （Multitask）No 
        classification_flag =hyper .get ("classification")
        if classification_flag is None :
            cls_tasks ={"mutagenicity","herg","bbbp","drd2","ames"}
            regression_tasks ={"esol","lipop","logs","logp","plogp","qed"}
            name_lower =task .lower ()
            if name_lower in cls_tasks :
                classification_flag =True 
            elif name_lower in regression_tasks :
                classification_flag =False 
            else :
                classification_flag =hyper .get ("regression_num",0 )==0 

                # If checkpoint MediumDimensionsandconsistent, Priority from checkpoint Inference hidden dimensions
        ffn_hidden_feats =hyper ["ffn_hidden_feats"]
        rgcn_hidden_feats =list (hyper ["rgcn_hidden_feats"])
        try :
            import torch as th 
            ckpt_dir =os .path .join (os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))),"checkpoints","sme")
            ckpt_path =os .path .join (ckpt_dir ,f"{task}_1_early_stop.pth")
            if os .path .exists (ckpt_path ):
                state =th .load (ckpt_path ,map_location ="cpu")["model_state_dict"]
                if "fc_layers1.1. weight"in state :
                    ffn_hidden_feats =state ["fc_layers1.1. weight"].shape [0 ]
                    # Inference RGCN hidden feats
                inferred =[]
                layer_idx =0 
                while True :
                    key =f"rgcn_gnn_layers. {layer_idx}. graph_conv_layer. h_bias"
                    if key in state :
                        inferred .append (state [key ].shape [0 ])
                        layer_idx +=1 
                    else :
                        break 
                if inferred :
                    rgcn_hidden_feats =inferred 
        except Exception :
            pass 

        for smi_col in tqdm (smiles_cols ,desc =f"{task} SMILES Columns",unit ="col",leave =False ):
            smis =df [smi_col ].astype (str ).tolist ()
            preds =SME_pred_for_mols (
            smis =smis ,
            model_name =task ,
            batch_size =args .batch_size ,
            rgcn_hidden_feats =rgcn_hidden_feats ,
            ffn_hidden_feats =ffn_hidden_feats ,
            lr =hyper ["lr"],
            classification =classification_flag ,
            device =args .device ,
            num_workers =args .num_workers ,
            num_seeds =args .num_seeds ,
            )

            col_name =f"{smi_col}_{task}_pred"
            out_df [col_name ]=preds 
            if classification_flag :
            # Category II: Threshold 0.5 Convert 0/1
                out_df [col_name ]=(out_df [col_name ]>=0.5 ).astype (int )

    out_df .to_csv (args .output_csv ,index =False )
    print (f"Projections，OutcomeSaveto: {args. output_csv}")


if __name__ =="__main__":
    main ()
