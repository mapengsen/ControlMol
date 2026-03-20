import argparse 
import math 
import os 
import shutil 
import subprocess 
import tempfile 
import warnings 
from collections import OrderedDict 
from pathlib import Path 
from typing import Dict ,List ,Optional ,Tuple ,Union 

import numpy as np 
import pandas as pd 
from rdkit import Chem ,DataStructs 
from rdkit .Chem import Descriptors ,MACCSkeys ,QED ,rdFingerprintGenerator ,rdMolDescriptors 

warnings .filterwarnings ("ignore")

from train_optimization_MMP .dataset import (
CLASSIFICATION_TOLERANCE ,
PROPERTY_CONFIG ,
calculate_binary_decoupling_score ,
sanitize_property_value ,
)

DEFAULT_PROPERTIES_TO_PRESERVE =["ESOL","hERG","lipop"]
RDKIT_PROPERTIES =["mw","qed","sa","logp","tpsa","hba","hbd","rob"]
DEFAULT_FINGERPRINT_TYPE ="morgan"
FINGERPRINT_TYPES =(DEFAULT_FINGERPRINT_TYPE ,"maccs")
MORGAN_RADIUS =2 
MORGAN_NBITS =2048 
MORGAN_GENERATOR =rdFingerprintGenerator .GetMorganGenerator (
radius =MORGAN_RADIUS ,fpSize =MORGAN_NBITS 
)


def _default_failed_output (output_csv :str )->Path :
    path =Path (output_csv )
    suffix =path .suffix or ".csv"
    return path .with_name (f"{path. stem}_unqualified{suffix}")


def _default_combined_output (output_csv :str )->Path :
    """Default path for saving (qualified + unqualified) molecules. """
    path =Path (output_csv )
    suffix =path .suffix or ".csv"
    return path .with_name (f"{path. stem}_with_unqualified{suffix}")


def _default_all_success_output (output_csv :str )->Path :
    """Default path for saving all successful candidates. """
    path =Path (output_csv )
    suffix =path .suffix or ".csv"
    return path .with_name (f"{path. stem}_all_success{suffix}")

def calculate_sa_score (smiles :str )->Optional [float ]:
    """Compute the synthetic accessibility score with a fallback implementation. """
    try :
        from rdkit .Contrib .SA_Score import sascorer 

        mol =Chem .MolFromSmiles (smiles )
        if mol is None :
            return None 
        return float (sascorer .calculateScore (mol ))
    except ImportError :
        mol =Chem .MolFromSmiles (smiles )
        if mol is None :
            return None 
        num_atoms =mol .GetNumAtoms ()
        num_bonds =mol .GetNumBonds ()
        num_rings =mol .GetRingInfo ().NumRings ()
        # Simple heuristic bounded to typical SA score range
        sa_estimate =num_atoms *0.1 +num_bonds *0.05 +num_rings *0.2 
        return float (min (sa_estimate ,10.0 ))


def compute_molecular_properties (smiles :str ,cache :Dict [str ,Dict [str ,float ]])->Optional [Dict [str ,float ]]:
    """Return cached molecular properties or compute them if missing. """
    if smiles in cache :
        return cache [smiles ]

    mol =Chem .MolFromSmiles (smiles )
    if mol is None :
        return None 

    try :
        mw =Descriptors .MolWt (mol )
        qed =QED .qed (mol )
        sa =calculate_sa_score (smiles )
        logp =Descriptors .MolLogP (mol )
        tpsa =Descriptors .TPSA (mol )
        hba =rdMolDescriptors .CalcNumHBA (mol )
        hbd =rdMolDescriptors .CalcNumHBD (mol )

        rob_violations =0 
        if mw >500 :
            rob_violations +=1 
        if logp >5 :
            rob_violations +=1 
        if hba >10 :
            rob_violations +=1 
        if hbd >5 :
            rob_violations +=1 

        properties ={
        "mw":float (mw ),
        "qed":float (qed ),
        "sa":None if sa is None else float (sa ),
        "logp":float (logp ),
        "tpsa":float (tpsa ),
        "hba":float (hba ),
        "hbd":float (hbd ),
        "rob":float (rob_violations ),
        }
    except Exception :
        return None 

    cache [smiles ]=properties 
    return properties 


def _get_fingerprint (mol :Chem .Mol ,fingerprint_type :str ):
    """Return a fingerprint object for the given molecule and type. """
    if fingerprint_type =="maccs":
        return MACCSkeys .GenMACCSKeys (mol )
    return MORGAN_GENERATOR .GetFingerprint (mol )


def compute_tanimoto_similarity (
smiles_a :str ,smiles_b :str ,fingerprint_type :str =DEFAULT_FINGERPRINT_TYPE 
)->Optional [float ]:
    """Compute Tanimoto similarity with selectable fingerprint (MorganDefault / MACCS Optional). """
    if not smiles_a or not smiles_b :
        return None 
    fp_type =(fingerprint_type or DEFAULT_FINGERPRINT_TYPE ).lower ()
    if fp_type not in FINGERPRINT_TYPES :
        fp_type =DEFAULT_FINGERPRINT_TYPE 
    mol_a =Chem .MolFromSmiles (smiles_a )
    mol_b =Chem .MolFromSmiles (smiles_b )
    if mol_a is None or mol_b is None :
        return None 
    fp_a =_get_fingerprint (mol_a ,fp_type )
    fp_b =_get_fingerprint (mol_b ,fp_type )
    return float (DataStructs .TanimotoSimilarity (fp_a ,fp_b ))


def resolve_candidate_similarity (
row_data :Dict [str ,object ],
similarity_column :Optional [str ],
source_smiles :str ,
optimized_smiles :str ,
min_similarity :float ,
fingerprint_type :str ,
)->Optional [float ]:
    """Return similarity from CSV column or compute it when needed. """
    similarity :Optional [float ]=None 
    if similarity_column and similarity_column in row_data :
        similarity =sanitize_numeric (row_data .get (similarity_column ))
    if similarity is None and (min_similarity >0 or not similarity_column ):
        similarity =compute_tanimoto_similarity (source_smiles ,optimized_smiles ,fingerprint_type )
    return similarity 


def sanitize_numeric (value :object )->Optional [float ]:
    """Convert value to finite float if possible. """
    if value is None :
        return None 
    if isinstance (value ,str ):
        value =value .strip ()
        if not value :
            return None 
    try :
        numeric =float (value )
    except (TypeError ,ValueError ):
        return None 
    if not np .isfinite (numeric ):
        return None 
    return float (numeric )


def extract_properties_from_row (
row_data :Dict [str ,object ],
prefix :Optional [str ],
suffix :str ,
properties :List [str ],
)->Dict [str ,float ]:
    """Grab property values from the CSV row using naming pattern. """
    if not prefix :
        return {}

    extracted :Dict [str ,float ]={}
    for prop in properties :
        column_name =f"{prefix}_{prop}{suffix}"
        if column_name not in row_data :
            continue 
        value =sanitize_numeric (row_data [column_name ])
        if value is not None :
            extracted [prop ]=value 
    return extracted 


def build_property_dict (
smiles :str ,
cache :Dict [str ,Dict [str ,float ]],
property_names :List [str ],
row_values :Optional [Dict [str ,float ]]=None ,
)->Optional [Dict [str ,float ]]:
    """Merge row-provided property values with RDKit descriptors as fallback. """
    properties :Dict [str ,float ]={}
    if row_values :
        properties .update (row_values )

    missing =[prop for prop in property_names if prop not in properties ]
    if missing :
        rdkit_props =compute_molecular_properties (smiles ,cache )
        if rdkit_props is None :
            return None 
        for prop in missing :
            if prop in rdkit_props :
                properties [prop ]=rdkit_props [prop ]

    if not properties :
        return None 

    for prop in property_names :
        if prop not in properties :
            return None 
    return properties 


def extract_expected_ratio (main_folder :Optional [str ])->Optional [float ]:
    """Parse the expected target ratio from folder naming (e. g. , inference_9_30_0.8). """
    if not main_folder :
        return None 
    normalized =main_folder .replace ("/","_").replace ("\\","_").replace ("-","_")
    tokens =normalized .split ("_")
    for token in reversed (tokens ):
        try :
            return float (token )
        except ValueError :
            continue 
    return None 


def compute_target_change_rate (
original_props :Dict [str ,float ],optimized_props :Dict [str ,float ],target_property :str 
)->Optional [float ]:
    start_val =original_props .get (target_property )
    end_val =optimized_props .get (target_property )

    if start_val is None or end_val is None :
        return None 
    if not np .isfinite (start_val )or not np .isfinite (end_val ):
        return None 
    if np .isclose (start_val ,0.0 ):
        return 0.0 
    return (end_val -start_val )/start_val 


def read_source_smiles (base_dir :str )->Optional [str ]:
    if not os .path .exists (base_dir ):
        return None 
    csv_path =os .path .join (base_dir ,"source_smiles.csv")
    if not os .path .exists (csv_path ):
        return None 
    df =pd .read_csv (csv_path )
    if df .empty or "source_smiles"not in df .columns :
        return None 
    return str (df .loc [0 ,"source_smiles"])


def _select_best_per_source (df :pd .DataFrame )->pd .DataFrame :
    if df .empty or "source_smiles"not in df .columns :
        return df 

    scores =pd .to_numeric (
    df .get ("decoupling_score",pd .Series (np .nan ,index =df .index )),errors ="coerce"
    ).fillna (np .inf )
    similarity =pd .to_numeric (
    df .get ("similarity",pd .Series (np .nan ,index =df .index )),errors ="coerce"
    ).fillna (-np .inf )
    sim_is_one =pd .Series (np .isclose (similarity .values ,1.0 ,atol =1e-9 ),index =df .index )
    confidence =pd .to_numeric (
    df .get ("confidence",pd .Series (np .nan ,index =df .index )),errors ="coerce"
    ).fillna (-np .inf )

    ordered =(
    df .assign (
    _sim_is_one =sim_is_one ,
    _score =scores ,
    _sim =similarity ,
    _conf =confidence ,
    )
    .sort_values (
    by =["source_smiles","_sim_is_one","_score","_sim","_conf"],
    ascending =[True ,True ,True ,False ,False ],
    kind ="mergesort",
    )
    )

    deduped =ordered .drop_duplicates (subset =["source_smiles"],keep ="first")
    # If in the end it is still similarity==1, Do not write best Outcome（The source Will）
    selected_sim =pd .to_numeric (deduped .get ("similarity",pd .Series (np .nan ,index =deduped .index )),errors ="coerce")
    selected_sim_is_one =np .isclose (selected_sim .fillna (-np .inf ).values ,1.0 ,atol =1e-9 )
    deduped =deduped .loc [~pd .Series (selected_sim_is_one ,index =deduped .index )]

    deduped =deduped .drop (columns =["_sim_is_one","_score","_sim","_conf"],errors ="ignore")
    deduped =deduped .reset_index (drop =True )
    return deduped 


def _resolve_required_delta (prop :str )->float :
    config =PROPERTY_CONFIG .get (prop ,{})
    required_delta =float (config .get ("preferred_delta",0.0 ))
    if abs (required_delta )<1e-12 :
        required_delta =0.5 
    return abs (required_delta )


def _is_target_property_success (
original_props :Dict [str ,float ],
optimized_props :Dict [str ,float ],
target_property :str ,
)->bool :
    start_val =sanitize_property_value (original_props .get (target_property ))
    end_val =sanitize_property_value (optimized_props .get (target_property ))
    config =PROPERTY_CONFIG .get (target_property ,{})
    prop_type =config .get ("type","regression")

    if prop_type =="classification":
        transition =config .get ("preferred_transition",{})
        tolerance =float (config .get ("classification_tolerance",CLASSIFICATION_TOLERANCE ))
        start_target =transition .get ("start")
        final_target =transition .get ("final")

        start_ok =True if start_target is None else np .isclose (start_val ,float (start_target ),atol =tolerance )
        final_ok =True if final_target is None else np .isclose (end_val ,float (final_target ),atol =tolerance )
        changed =not np .isclose (end_val ,start_val ,atol =tolerance )
        return start_ok and final_ok and changed 

    required_delta =float (config .get ("preferred_delta",0.0 ))
    delta =end_val -start_val 
    if required_delta >=0 :
        return delta >=(required_delta -1e-9 )
    return delta <=(required_delta +1e-9 )


def _are_preserved_properties_stable (
original_props :Dict [str ,float ],
optimized_props :Dict [str ,float ],
properties_to_preserve :List [str ],
)->bool :
    for prop in properties_to_preserve :
        start_val =sanitize_property_value (original_props .get (prop ))
        end_val =sanitize_property_value (optimized_props .get (prop ))
        config =PROPERTY_CONFIG .get (prop ,{})
        if config .get ("type")=="classification":
            tolerance =float (config .get ("classification_tolerance",CLASSIFICATION_TOLERANCE ))
        else :
            tolerance =float (config .get ("regression_threshold",0.5 ))
        if not np .isclose (end_val ,start_val ,atol =tolerance ):
            return False 
    return True 


def _parse_prediction_columns (config :Optional [str ])->"OrderedDict[str, str]":
    """Parse extra prediction column mappings specified as 'column' or 'column: prefix'. """
    mapping :"OrderedDict[str, str]"=OrderedDict ()
    if not config :
        return mapping 

    for item in config .split (", "):
        item =item .strip ()
        if not item :
            continue 
        if ": "in item :
            column ,prefix =item .split (": ",1 )
            column =column .strip ()
            prefix =prefix .strip ()
        else :
            column =item 
            prefix =item 
        if not column or not prefix :
            continue 
        mapping [column ]=prefix 
    return mapping 


def _collect_prediction_column_map (
optimized_column :Optional [str ],
optimized_prefix :Optional [str ],
source_column :Optional [str ],
source_prefix :Optional [str ],
extra_mapping :Optional [str ],
)->"OrderedDict[str, str]":
    """Build ordered mapping between SMILES columns and their property column prefixes. """
    column_map :"OrderedDict[str, str]"=OrderedDict ()
    if optimized_column :
        column_map [optimized_column ]=optimized_prefix or optimized_column 
    if source_column :
        column_map [source_column ]=source_prefix or source_column 
    column_map .update (_parse_prediction_columns (extra_mapping ))
    return column_map 


def _find_missing_prediction_columns (
df :pd .DataFrame ,
column_prefix_map :"OrderedDict[str, str]",
property_suffix :str ,
properties :List [str ],
)->Dict [str ,List [str ]]:
    """Return mapping of SMILES columns to missing property columns. """
    missing :Dict [str ,List [str ]]={}
    for smiles_col ,prefix in column_prefix_map .items ():
        expected =[f"{prefix}_{prop}{property_suffix}"for prop in properties ]
        absent =[col for col in expected if col not in df .columns ]
        if absent :
            missing [smiles_col ]=absent 
    return missing 


def run_property_prediction_if_needed (
df :pd .DataFrame ,
input_csv_path :Path ,
column_prefix_map :"OrderedDict[str, str]",
properties :List [str ],
property_suffix :str ,
checkpoint_root :Optional [str ],
batch_size :int ,
chunk_size :int ,
num_workers :int ,
gpu :Optional [int ],
force :bool ,
skip :bool ,
)->Tuple [pd .DataFrame ,bool ]:
    """Run integrated property prediction using KANO models when required. """
    prediction_performed =False 
    if not column_prefix_map or not properties :
        return df ,prediction_performed 

    missing_targets =_find_missing_prediction_columns (df ,column_prefix_map ,property_suffix ,properties )
    need_prediction =force or any (missing_targets .values ())
    if not need_prediction :
        return df ,prediction_performed 

    if skip :
        if missing_targets :
            for smiles_col ,absent in missing_targets .items ():
                print (f"Skipping property prediction: column {smiles_col} is missing {absent}")
        return df ,prediction_performed 

    if not checkpoint_root :
        raise ValueError ("A property prediction checkpoint root is required for automatic prediction. ")

    checkpoint_root_path =Path (checkpoint_root )
    if not checkpoint_root_path .exists ():
        raise FileNotFoundError (f"Property prediction checkpoint root does not exist: {checkpoint_root_path}")

    missing_smiles_cols =[col for col in column_prefix_map if col not in df .columns ]
    if missing_smiles_cols :
        missing_str =", ".join (missing_smiles_cols )
        raise KeyError (f"InputDataMissingBelowSMILESColumns: {missing_str}")

    conda_exec =shutil .which ("conda")
    if not conda_exec :
        raise EnvironmentError ("Could not find the conda executable required to run property prediction in the mocov3 environment. ")

    script_path =Path (__file__ ).resolve ().parent /"predictModel"/"KANO"/"predict_pair_properties.py"
    if not script_path .exists ():
        raise FileNotFoundError (f"Could not locate the property prediction script: {script_path}")

    columns_to_predict =list (column_prefix_map .keys ())
    if not columns_to_predict :
        return df ,prediction_performed 

    input_path =Path (input_csv_path )
    if not input_path .exists ():
        raise FileNotFoundError (f"Could not find the input CSV file: {input_path}")

    chunk_size =int (chunk_size )
    num_workers =int (num_workers )

    with tempfile .NamedTemporaryFile (prefix ="kano_predictions_",suffix =".csv",delete =False )as tmp_file :
        temp_output_path =Path (tmp_file .name )

    command :List [str ]=[
    conda_exec ,
    "run",
    "-n",
    "mocov3",
    "python",
    str (script_path ),
    "--input_path",
    str (input_path ),
    "--output_path",
    str (temp_output_path ),
    "--checkpoint_root",
    str (checkpoint_root_path ),
    "--batch_size",
    str (batch_size ),
    "--chunk_size",
    str (chunk_size ),
    "--num_workers",
    str (num_workers ),
    ]

    if gpu is not None :
        command .extend (["--gpu",str (gpu )])

    command .append ("--columns")
    command .extend (columns_to_predict )
    command .append ("--tasks")
    command .extend (properties )

    print ("[Property Prediction] Running predict_pair_properties.py in the mocov3 environment")

    try :
        subprocess .run (command ,check =True )
        predictions_df =pd .read_csv (temp_output_path )
        if len (predictions_df )!=len (df ):
            raise ValueError ("The prediction output row count does not match the input CSV. Please check the prediction script output. ")
        for smiles_col ,prefix in column_prefix_map .items ():
            raw_prefix =f"{smiles_col}_"
            for prop in properties :
                raw_col =f"{raw_prefix}{prop}"
                target_col =f"{prefix}_{prop}{property_suffix}"
                if raw_col not in predictions_df .columns :
                    print (f"Warning: prediction output is missing column {raw_col}; skipping write. ")
                    continue 
                df [target_col ]=predictions_df [raw_col ].values 
        prediction_performed =True 
    except subprocess .CalledProcessError as exc :
        raise RuntimeError (f"Property prediction via the mocov3 environment failed: {exc}")from exc 
    finally :
        try :
            temp_output_path .unlink ()
        except FileNotFoundError :
            pass 

    return df ,prediction_performed 


def process_groups (
df :pd .DataFrame ,
target_property :str ,
properties_to_preserve :List [str ],
property_suffix :str ,
optimized_smiles_column :str ,
source_smiles_column :Optional [str ],
optimized_property_prefix :Optional [str ],
source_property_prefix :Optional [str ],
similarity_column :Optional [str ],
min_similarity :float ,
fingerprint_type :str ,
collect_all_success :bool =False ,
)->Union [Tuple [pd .DataFrame ,pd .DataFrame ],Tuple [pd .DataFrame ,pd .DataFrame ,pd .DataFrame ]]:
    property_cache :Dict [str ,Dict [str ,float ]]={}
    selected_success :List [Dict [str ,object ]]=[]
    selected_failed :List [Dict [str ,object ]]=[]
    all_success_records :List [Dict [str ,object ]]=[]
    all_properties =list (dict .fromkeys ([target_property ,*properties_to_preserve ]))
    optimized_prefix =optimized_property_prefix or optimized_smiles_column 
    source_prefix =source_property_prefix or source_smiles_column 

    if "main_folder"in df .columns and "subfolder_path"in df .columns :
        group_cols =["main_folder","subfolder_path"]
        if source_smiles_column and source_smiles_column in df .columns :
            group_cols .append (source_smiles_column )
        grouped =df .groupby (group_cols ,sort =False )
    else :
        grouped =[((None ,None ,None ),df )]

    def _failed_priority (entry :Dict [str ,object ])->Tuple [float ,float ,float ]:
        """Prefer failed candidates with higher similarity and better scores. """
        sim =sanitize_numeric (entry .get ("similarity"))
        sim_val =sim if sim is not None and np .isfinite (sim )else -np .inf 
        score_val =sanitize_numeric (entry .get ("decoupling_score"))
        score_val =score_val if score_val is not None and np .isfinite (score_val )else -np .inf 
        return (
        1.0 if entry .get ("similarity_passed")else 0.0 ,
        sim_val ,
        score_val ,
        )

    for group_key ,group in grouped :
        if not isinstance (group_key ,tuple ):
            group_key =(group_key ,)
        main_folder =group_key [0 ]if len (group_key )>0 else None 
        subfolder =group_key [1 ]if len (group_key )>1 else None 
        expected_ratio =extract_expected_ratio (main_folder )
        best_candidate =None 
        best_score =None 
        failed_candidates :List [Tuple [Tuple [float ,float ,float ],Dict [str ,object ]]]=[]

        for row in group .itertuples (index =False ):
            row_dict =row ._asdict ()

            optimized_smiles =row_dict .get (optimized_smiles_column )
            if not isinstance (optimized_smiles ,str )or not optimized_smiles :
                continue 
            source_smiles =(
            row_dict .get (source_smiles_column )
            if source_smiles_column and source_smiles_column in row_dict 
            else None 
            )
            if not isinstance (source_smiles ,str )or not source_smiles :
                continue 

            similarity =resolve_candidate_similarity (
            row_dict ,
            similarity_column ,
            source_smiles ,
            optimized_smiles ,
            min_similarity ,
            fingerprint_type ,
            )

            optimized_row_props =extract_properties_from_row (
            row_dict ,optimized_prefix ,property_suffix ,all_properties 
            )
            original_row_props =extract_properties_from_row (
            row_dict ,source_prefix ,property_suffix ,all_properties 
            )

            optimized_props =build_property_dict (
            optimized_smiles ,property_cache ,all_properties ,optimized_row_props 
            )
            if optimized_props is None :
                continue 

            original_props =build_property_dict (
            source_smiles ,property_cache ,all_properties ,original_row_props 
            )
            if original_props is None :
                continue 

            target_success =_is_target_property_success (original_props ,optimized_props ,target_property )
            preserved_stable =_are_preserved_properties_stable (original_props ,optimized_props ,properties_to_preserve )

            score =calculate_binary_decoupling_score (
            original_props ,
            optimized_props ,
            target_property =target_property ,
            properties_to_preserve =properties_to_preserve ,
            )

            change_rate =compute_target_change_rate (original_props ,optimized_props ,target_property )

            ratio =math .nan 
            original_target =sanitize_property_value (original_props .get (target_property ))
            optimized_target =sanitize_property_value (optimized_props .get (target_property ))
            if not math .isclose (original_target ,0.0 ,abs_tol =1e-12 ):
                ratio =optimized_target /original_target 
            elif math .isclose (optimized_target ,0.0 ,abs_tol =1e-12 ):
                ratio =1.0 

            if expected_ratio is None :
                ratio_deviation =0.0 if not math .isnan (ratio )else math .inf 
            else :
                ratio_deviation =math .inf if math .isnan (ratio )else abs (ratio -expected_ratio )

            similarity_passed =not (min_similarity >0 and (similarity is None or similarity <min_similarity ))
            candidate_success =(
            similarity_passed 
            and target_success 
            and preserved_stable 
            and score is not None 
            )

            failure_reasons :List [str ]=[]
            if not similarity_passed :
                failure_reasons .append ("similarity_below_threshold")
            if not target_success :
                failure_reasons .append ("target_not_changed")
            if not preserved_stable :
                failure_reasons .append ("preserve_changed")
            if score is None :
                failure_reasons .append ("score_nan")

            candidate_record ={
            "main_folder":main_folder ,
            "subfolder_path":subfolder ,
            "source_smiles":source_smiles ,
            "optimized_smiles":optimized_smiles ,
            "decoupling_score":score if score is not None else math .nan ,
            "target_change_rate":change_rate if change_rate is not None else math .nan ,
            "target_ratio":ratio ,
            "expected_ratio":expected_ratio ,
            "ratio_deviation":ratio_deviation ,
            "decoupling_success":candidate_success ,
            "similarity_passed":similarity_passed ,
            "failure_reason":""if candidate_success else "; ".join (failure_reasons ),
            "confidence":getattr (row ,"confidence",np .nan ),
            "image_name":getattr (row ,"image_name",""),
            "similarity":similarity if similarity is not None else math .nan ,
            "optimized_properties":optimized_props ,
            "original_properties":original_props ,
            }

            if candidate_success :
                if collect_all_success :
                    success_record =dict (candidate_record )
                    for prop in all_properties :
                        success_record [f"optimized_{prop}"]=optimized_props .get (prop ,math .nan )
                        success_record [f"original_{prop}"]=original_props .get (prop ,math .nan )
                    success_record .pop ("optimized_properties",None )
                    success_record .pop ("original_properties",None )
                    all_success_records .append (success_record )

                if best_candidate is None :
                    best_candidate =candidate_record 
                    best_score =score 
                else :
                    update =False 
                    prev_sim_num =sanitize_numeric (best_candidate .get ("similarity",np .nan ))if best_candidate else None 
                    prev_sim_is_one =(
                    prev_sim_num is not None 
                    and np .isfinite (prev_sim_num )
                    and np .isclose (prev_sim_num ,1.0 ,atol =1e-9 )
                    )
                    current_sim_is_one =(
                    similarity is not None and np .isfinite (similarity )and np .isclose (similarity ,1.0 ,atol =1e-9 )
                    )

                    # Prefer candidates with similarity != 1 when both are successful.
                    # A similarity of 1 usually means the structure did not actually change.
                    if prev_sim_is_one !=current_sim_is_one :
                        update =prev_sim_is_one and (not current_sim_is_one )
                    else :
                        if score is not None and best_score is not None and score <best_score -1e-9 :
                            update =True 
                        elif score is not None and best_score is not None and abs (score -best_score )<=1e-9 :
                            prev_sim_raw =best_candidate .get ("similarity",np .nan )if best_candidate else np .nan 
                            prev_sim_num =sanitize_numeric (prev_sim_raw )
                            current_sim_num =similarity if similarity is not None and np .isfinite (similarity )else None 
                            if current_sim_num is not None and prev_sim_num is None :
                                update =True 
                            elif current_sim_num is None and prev_sim_num is not None :
                                update =False 
                            elif current_sim_num is not None and prev_sim_num is not None :
                                if current_sim_num >prev_sim_num +1e-9 :
                                    update =True 
                                elif abs (current_sim_num -prev_sim_num )<=1e-9 :
                                    current_conf =getattr (row ,"confidence",np .nan )
                                    prev_conf =best_candidate .get ("confidence",np .nan )if best_candidate else np .nan 
                                    if np .isfinite (current_conf )and (
                                    not np .isfinite (prev_conf )or current_conf >prev_conf +1e-9 
                                    ):
                                        update =True 
                            else :
                                current_conf =getattr (row ,"confidence",np .nan )
                                prev_conf =best_candidate .get ("confidence",np .nan )if best_candidate else np .nan 
                                if np .isfinite (current_conf )and (
                                not np .isfinite (prev_conf )or current_conf >prev_conf +1e-9 
                                ):
                                    update =True 

                    if update :
                        best_candidate .update (
                        {
                        "optimized_smiles":optimized_smiles ,
                        "decoupling_score":score if score is not None else math .nan ,
                        "target_change_rate":change_rate if change_rate is not None else math .nan ,
                        "target_ratio":ratio ,
                        "expected_ratio":expected_ratio ,
                        "ratio_deviation":ratio_deviation ,
                        "decoupling_success":True ,
                        "confidence":getattr (row ,"confidence",np .nan ),
                        "image_name":getattr (row ,"image_name",""),
                        "similarity":similarity if similarity is not None else math .nan ,
                        "optimized_properties":optimized_props ,
                        }
                        )
                        best_score =score 
            else :
                candidate_key =_failed_priority (candidate_record )
                failed_candidates .append ((candidate_key ,candidate_record ))

        if best_candidate is not None :
            optimized_props =best_candidate .pop ("optimized_properties")
            original_props =best_candidate .pop ("original_properties")

            record ={**best_candidate }
            for prop in all_properties :
                record [f"optimized_{prop}"]=optimized_props .get (prop ,math .nan )
                record [f"original_{prop}"]=original_props .get (prop ,math .nan )

            selected_success .append (record )

        if best_candidate is None and failed_candidates :
            failed_candidates .sort (key =lambda item :item [0 ],reverse =True )
            select_index =0 
            if len (failed_candidates )>=2 :
                top_candidate =failed_candidates [0 ][1 ]
                top_similarity =sanitize_numeric (top_candidate .get ("similarity"))
                if top_similarity is not None and np .isclose (top_similarity ,1.0 ,atol =1e-9 ):
                    select_index =1 

            failed_candidate =failed_candidates [select_index ][1 ]
            optimized_props =failed_candidate .pop ("optimized_properties")
            original_props =failed_candidate .pop ("original_properties")

            record ={**failed_candidate }
            for prop in all_properties :
                record [f"optimized_{prop}"]=optimized_props .get (prop ,math .nan )
                record [f"original_{prop}"]=original_props .get (prop ,math .nan )

            selected_failed .append (record )

    success_df =pd .DataFrame (selected_success )
    failed_df =pd .DataFrame (selected_failed )
    if collect_all_success :
        all_success_df =pd .DataFrame (all_success_records )
        return _select_best_per_source (success_df ),failed_df ,all_success_df 
    return _select_best_per_source (success_df ),failed_df 


def parse_args ()->argparse .Namespace :
    parser =argparse .ArgumentParser (description ="Select best decoupled molecules from inference results. ")
    parser .add_argument (
    "--input_csv",
    default ="results/molecular_smiles_detailed_results.csv",
    help ="Path to the detailed inference CSV containing candidate molecules. ",
    )
    parser .add_argument (
    "--output_csv",
    default ="results/best_decoupled_smiles.csv",
    help ="Path to store the selected best molecules. ",
    )
    parser .add_argument (
    "--target_property",
    default ="mw",
    help ="Target property optimized during training (default: mw). ",
    )
    parser .add_argument (
    "--properties_to_preserve",
    default =", ".join (DEFAULT_PROPERTIES_TO_PRESERVE ),
    help ="Comma-separated list of properties to preserve when evaluating decoupling. ",
    )
    parser .add_argument (
    "--optimized_smiles_column",
    default ="smiles",
    help ="Column name for optimized SMILES in the input CSV. ",
    )
    parser .add_argument (
    "--source_smiles_column",
    default =None ,
    help ="Column name for source SMILES in the input CSV. "
    "If not provided, the script will rely on other columns or upstream steps to supply source molecule information. ",
    )
    parser .add_argument (
    "--optimized_property_prefix",
    default =None ,
    help ="Prefix for optimized property columns (e. g. 'smiles'). "
    "Defaults to optimized_smiles_column. ",
    )
    parser .add_argument (
    "--source_property_prefix",
    default =None ,
    help ="Prefix for source property columns (e. g. 'source_smiles'). "
    "Defaults to source_smiles_column. ",
    )
    parser .add_argument (
    "--property_suffix",
    default ="_pred",
    help ="Suffix appended to property columns (e. g. '_pred'). ",
    )
    parser .add_argument (
    "--prediction_checkpoint_root",
    default ="predictModel/KANO/dumped/finetune",
    help ="Root directory for KANO property predictors. Leave empty to disable automatic prediction. ",
    )
    parser .add_argument (
    "--prediction_columns",
    default =None ,
    help ="Additional SMILES columns to predict. Format: column or column: prefix, separated by commas. ",
    )
    parser .add_argument (
    "--prediction_batch_size",
    type =int ,
    default =512 ,
    help ="Batch size used during property prediction. ",
    )
    parser .add_argument (
    "--prediction_chunk_size",
    type =int ,
    default =100000 ,
    help ="chunk_size passed to predict_pair_properties.py. ",
    )
    parser .add_argument (
    "--prediction_num_workers",
    type =int ,
    default =-1 ,
    help ="num_workers passed to predict_pair_properties.py. ",
    )
    parser .add_argument (
    "--prediction_gpu",
    type =int ,
    default =None ,
    help ="GPU index for property prediction; leave empty or use a negative value for CPU. ",
    )
    parser .add_argument (
    "--similarity_column",
    default ="similarity",
    help ="Name of the Tanimoto similarity column in the input CSV; leave empty to compute it on the fly. ",
    )
    parser .add_argument (
    "--min_similarity",
    type =float ,
    default =0.4 ,
    help ="Minimum Tanimoto similarity required when filtering candidates. ",
    )
    parser .add_argument (
    "--similarity_fingerprint",
    choices =FINGERPRINT_TYPES ,
    default =DEFAULT_FINGERPRINT_TYPE ,
    help ="Fingerprint type used to compute Tanimoto similarity: morgan or maccs. ",
    )
    parser .add_argument (
    "--skip_property_prediction",
    action ="store_true",
    help ="Skip the integrated property prediction step. ",
    )
    parser .add_argument (
    "--force_property_prediction",
    action ="store_true",
    help ="Force property prediction even if prediction columns already exist. ",
    )
    parser .add_argument (
    "--failed_output_csv",
    default =None ,
    help ="When no successful decoupled candidate is found, save fallback candidates here. Defaults to output_csv with a suffix. ",
    )
    parser .add_argument (
    "--combined_output_csv",
    default =None ,
    help ="Optional CSV path for the combined output of successful and fallback candidates. Defaults to output_csv with a suffix. ",
    )
    parser .add_argument (
    "--all_success_output_csv",
    default =None ,
    help ="Optional CSV path for all successful decoupled candidates. Defaults to output_csv with a suffix. ",
    )
    return parser .parse_args ()


def main ()->None :
    args =parse_args ()

    df =pd .read_csv (args .input_csv )
    if df .empty :
        print ("Input CSV is empty; nothing to process. ")
        return 
    properties_to_preserve =[
    prop .strip ()for prop in args .properties_to_preserve .split (", ")if prop .strip ()
    ]
    if not properties_to_preserve :
        print ("No properties to preserve were provided; cannot compute decoupling scores. ")
        return 
    all_properties =list (dict .fromkeys ([args .target_property ,*properties_to_preserve ]))

    column_prefix_map =_collect_prediction_column_map (
    args .optimized_smiles_column ,
    args .optimized_property_prefix ,
    args .source_smiles_column ,
    args .source_property_prefix ,
    args .prediction_columns ,
    )

    checkpoint_root =args .prediction_checkpoint_root .strip ()if args .prediction_checkpoint_root else None 
    df ,prediction_performed =run_property_prediction_if_needed (
    df ,
    Path (args .input_csv ),
    column_prefix_map ,
    all_properties ,
    args .property_suffix ,
    checkpoint_root ,
    args .prediction_batch_size ,
    args .prediction_chunk_size ,
    args .prediction_num_workers ,
    args .prediction_gpu ,
    args .force_property_prediction ,
    args .skip_property_prediction ,
    )

    if prediction_performed :
        try :
            df .to_csv (args .input_csv ,index =False )
            print (f"Saved the property-augmented input CSV to {args. input_csv}")
        except Exception as exc :
            print (f"Warning: could not save the property-augmented input CSV {args. input_csv}: {exc}")

    success_df ,failed_df ,all_success_df =process_groups (
    df ,
    target_property =args .target_property ,
    properties_to_preserve =properties_to_preserve ,
    property_suffix =args .property_suffix ,
    optimized_smiles_column =args .optimized_smiles_column ,
    source_smiles_column =args .source_smiles_column ,
    optimized_property_prefix =args .optimized_property_prefix ,
    source_property_prefix =args .source_property_prefix ,
    similarity_column =args .similarity_column ,
    min_similarity =args .min_similarity ,
    fingerprint_type =args .similarity_fingerprint ,
    collect_all_success =True ,
    )

    success_count =len (success_df .index )
    denominator =None 
    if args .source_smiles_column and args .source_smiles_column in df .columns :
        denominator =df [args .source_smiles_column ].dropna ().nunique ()
    elif "main_folder"in df .columns :
        denominator =df ["main_folder"].dropna ().nunique ()

    if denominator :
        success_rate =success_count /denominator 
        print (f"Decoupling success rate: {success_count}/{denominator} = {success_rate: .2%}")
    else :
        print ("Warning: could not determine the denominator for success rate (missing source_smiles or main_folder columns). ")

    output_path =Path (args .output_csv )
    output_path .parent .mkdir (parents =True ,exist_ok =True )

    if success_df .empty :
        print ("No optimized results passed selection. ")
        success_df .to_csv (output_path ,index =False )
        print (f"Saved empty result file to {output_path}")
    else :
        success_df .to_csv (output_path ,index =False )
        print (f"Saved selected results to {output_path}")

    failed_output_path =Path (args .failed_output_csv )if args .failed_output_csv else _default_failed_output (output_path )
    if not failed_df .empty :
        failed_output_path .parent .mkdir (parents =True ,exist_ok =True )
        failed_df .to_csv (failed_output_path ,index =False )
        print (f"Saved unqualified candidates to {failed_output_path}")
    else :
        print ("No unqualified candidates found. ")

    combined_output_csv =args .combined_output_csv 
    combined_output_path :Optional [Path ]=None 
    if combined_output_csv is None :
        combined_output_path =_default_combined_output (output_path )
    elif isinstance (combined_output_csv ,str )and combined_output_csv .strip ():
        combined_output_path =Path (combined_output_csv .strip ())

    success_pool =all_success_df if not all_success_df .empty else success_df 
    all_success_output_csv =args .all_success_output_csv 
    all_success_output_path :Optional [Path ]=None 
    if all_success_output_csv is None :
        all_success_output_path =_default_all_success_output (output_path )
    elif isinstance (all_success_output_csv ,str )and all_success_output_csv .strip ():
        all_success_output_path =Path (all_success_output_csv .strip ())

    if all_success_output_path is not None :
        all_success_output_path .parent .mkdir (parents =True ,exist_ok =True )
        if success_pool .empty :
            success_pool .to_csv (all_success_output_path ,index =False )
            print ("No successful decoupled candidates found. ")
            print (f"Saved empty successful-candidate CSV to {all_success_output_path}")
        else :
            success_pool .to_csv (all_success_output_path ,index =False )
            print (f"Saved all successful decoupled candidates to {all_success_output_path}")

    if combined_output_path is not None :
        combined_output_path .parent .mkdir (parents =True ,exist_ok =True )
        combined_df =pd .concat ([success_pool ,failed_df ],ignore_index =True ,sort =False )
        combined_df .to_csv (combined_output_path ,index =False )
        print (f"Saved combined successful + unqualified candidates to {combined_output_path}")


if __name__ =="__main__":
    main ()
