import torch 
from torch .utils .data import Dataset ,DataLoader 
import pandas as pd 
import numpy as np 
from rdkit import Chem 
from rdkit .Chem import Draw 
from PIL import Image 
import os 
import json 
import torchvision .transforms as transforms 
import pretrained_enc .models_pretrained_enc as models_pretrained_enc 
from typing import Dict ,List ,Optional ,Tuple 
from collections import defaultdict 
from tqdm import tqdm 
import time 
import random 

PROPERTY_CONFIG :Dict [str ,Dict [str ,object ]]={
'B1':{
'type':'classification',
'preferred_transition':{'start':0.0 ,'final':1.0 }
},
'B2':{
'type':'classification',
'preferred_transition':{'start':1.0 ,'final':0.0 }
},
'EPHX2':{
'type':'classification',
'preferred_transition':{'start':0.0 ,'final':1.0 }
},
'EP4':{
'type':'classification',
'preferred_transition':{'start':0.0 ,'final':1.0 }
},
'EP2':{
'type':'classification',
'preferred_transition':{'start':0.0 ,'final':1.0 }
},
'BBBP':{
'type':'classification',
'preferred_transition':{'start':0.0 ,'final':1.0 }
},
'ESOL':{
'type':'regression',
'regression_threshold':0.5 ,
'preferred_delta':0.5 
},
'hERG':{
'type':'classification',
'preferred_transition':{'start':1.0 ,'final':0.0 }
},
'lipop':{
'type':'regression',
'regression_threshold':0.5 ,
'preferred_delta':0.5 
},
'Mutagenicity':{
'type':'classification',
'preferred_transition':{'start':1.0 ,'final':0.0 }
},
'mw':{
'type':'regression',
'regression_threshold':0.1 ,
'preferred_delta':0.0 
},
'qed':{
'type':'regression',
'regression_threshold':0.1 ,
'preferred_delta':0.0 
},
'sa':{
'type':'regression',
'regression_threshold':0.1 ,
'preferred_delta':0.0 
},
'logp':{
'type':'regression',
'regression_threshold':0.1 ,
'preferred_delta':0.0 
},
'tpsa':{
'type':'regression',
'regression_threshold':0.1 ,
'preferred_delta':0.0 
},
'hba':{
'type':'regression',
'regression_threshold':0.1 ,
'preferred_delta':0.0 
},
'hbd':{
'type':'regression',
'regression_threshold':0.1 ,
'preferred_delta':0.0 
},
'rob':{
'type':'regression',
'regression_threshold':0.1 ,
'preferred_delta':0.0 
},
}

CLASSIFICATION_TOLERANCE =0.0 


def sanitize_property_value (value :object )->float :
    """Convert arbitrary value to finite float with training-consistent fallback. """
    try :
        numeric =float (value )
    except (TypeError ,ValueError ):
        return 0.0 
    if not np .isfinite (numeric ):
        return 0.0 
    return float (numeric )


def _classification_success (
start_vals :np .ndarray ,
final_vals :np .ndarray ,
transition :Optional [Dict [str ,float ]],
tolerance :float =CLASSIFICATION_TOLERANCE ,
)->np .ndarray :
    start_target =None if transition is None else transition .get ('start')
    final_target =None if transition is None else transition .get ('final')

    start_ok =np .ones_like (start_vals ,dtype =bool )
    if start_target is not None :
        start_ok =np .isclose (start_vals ,float (start_target ),atol =tolerance )

    final_ok =np .ones_like (final_vals ,dtype =bool )
    if final_target is not None :
        final_ok =np .isclose (final_vals ,float (final_target ),atol =tolerance )

    return start_ok &final_ok 


def _classification_success_scalar (
start_val :float ,
final_val :float ,
transition :Optional [Dict [str ,float ]],
tolerance :float =CLASSIFICATION_TOLERANCE ,
)->bool :
    start_target =None if transition is None else transition .get ('start')
    final_target =None if transition is None else transition .get ('final')

    start_ok =True 
    if start_target is not None :
        start_ok =abs (start_val -float (start_target ))<=tolerance 

    final_ok =True 
    if final_target is not None :
        final_ok =abs (final_val -float (final_target ))<=tolerance 

    return start_ok and final_ok 


def _binary_penalty_array (
start_vals :np .ndarray ,
final_vals :np .ndarray ,
prop :str ,
is_target :bool =False ,
)->np .ndarray :
    config =PROPERTY_CONFIG .get (prop ,{})
    prop_type =config .get ('type','regression')

    if is_target :
        if prop_type =='classification':
            transition =config .get ('preferred_transition')
            success =_classification_success (start_vals ,final_vals ,transition )
            penalty =np .where (success ,0.0 ,1.0 )
        else :
            required_delta =float (config .get ('preferred_delta',0.0 ))
            success =(final_vals -start_vals )>=required_delta 
            penalty =np .where (success ,0.0 ,1.0 )
    else :
        if prop_type =='classification':
            tolerance =float (config .get ('classification_tolerance',CLASSIFICATION_TOLERANCE ))
            failure =~np .isclose (final_vals ,start_vals ,atol =tolerance )
            penalty =failure .astype (np .float32 )
        else :
            threshold =float (config .get ('regression_threshold',0.5 ))
            diff =np .abs (final_vals -start_vals )
            penalty =(diff >=threshold ).astype (np .float32 )

    return penalty .astype (np .float32 )


def _binary_penalty_scalar (
start_val :float ,
final_val :float ,
prop :str ,
is_target :bool =False ,
)->float :
    config =PROPERTY_CONFIG .get (prop ,{})
    prop_type =config .get ('type','regression')

    if is_target :
        if prop_type =='classification':
            transition =config .get ('preferred_transition')
            success =_classification_success_scalar (start_val ,final_val ,transition )
            return 0.0 if success else 1.0 
        else :
            required_delta =float (config .get ('preferred_delta',0.0 ))
            return 0.0 if (final_val -start_val )>=required_delta else 1.0 
    else :
        if prop_type =='classification':
            tolerance =float (config .get ('classification_tolerance',CLASSIFICATION_TOLERANCE ))
            return 0.0 if abs (final_val -start_val )<=tolerance else 1.0 
        else :
            threshold =float (config .get ('regression_threshold',0.5 ))
            return 1.0 if abs (final_val -start_val )>=threshold else 0.0 


def calculate_binary_decoupling_score (
original_props :Dict [str ,float ],
optimized_props :Dict [str ,float ],
target_property :str ,
properties_to_preserve :List [str ],
)->Optional [float ]:
    """Compute the binary decoupling score for a single sample. """
    all_props =[target_property ,*properties_to_preserve ]
    total_properties =len (all_props )
    if total_properties ==0 :
        return None 

    total_penalty =0.0 
    for prop in all_props :
        original =sanitize_property_value (original_props .get (prop ))
        optimized =sanitize_property_value (optimized_props .get (prop ))
        penalty =_binary_penalty_scalar (
        original ,
        optimized ,
        prop ,
        is_target =(prop ==target_property ),
        )
        total_penalty +=penalty 

    return float (total_penalty /total_properties )


def is_binary_decoupling_success (
original_props :Dict [str ,float ],
optimized_props :Dict [str ,float ],
target_property :str ,
properties_to_preserve :List [str ],
)->bool :
    """Return True if target changed while preserve properties stayed stable. """
    target_start =sanitize_property_value (original_props .get (target_property ))
    target_end =sanitize_property_value (optimized_props .get (target_property ))
    target_changed =not np .isclose (target_end ,target_start ,atol =1e-6 )
    if not target_changed :
        return False 

    for prop in properties_to_preserve :
        start_val =sanitize_property_value (original_props .get (prop ))
        end_val =sanitize_property_value (optimized_props .get (prop ))
        config =PROPERTY_CONFIG .get (prop ,{})
        if config .get ('type')=='classification':
            tolerance =float (config .get ('classification_tolerance',CLASSIFICATION_TOLERANCE ))
        else :
            tolerance =float (config .get ('regression_threshold',0.5 ))
        if not np .isclose (end_val ,start_val ,atol =tolerance ):
            return False 

    return True 


def scale_to_range (x ):
    """Will[0, 1]Zoom to[-1, 1]"""
    return x *2.0 -1.0 


def normalize_representation (rep ):
    """anddiffusionModelTrainingconsistentZ-scoreStandardization"""
    rep_std =torch .std (rep ,dim =1 ,keepdim =True )
    rep_mean =torch .mean (rep ,dim =1 ,keepdim =True )
    rep_std =torch .clamp (rep_std ,min =1e-8 )# Avoiding the elimination of zeroes
    return (rep -rep_mean )/rep_std 


class MMPDataset (Dataset ):
    """MMPData，ForTraining"""

    def __init__ (
    self ,
    csv_path :str ,
    encoder_model =None ,
    target_property :str ='mw',
    property_list :Optional [List [str ]]=None ,
    property_label_column :Optional [str ]=None ,
    filter_by_property_label :bool =False ,
    transform =None ,
    negative_sample_size :int =5 ,
    properties_to_preserve :List [str ]=None ,
    correlation_file :Optional [str ]=None ,
    deviation_threshold :float =0.1 ,
    batch_size :int =32 ,
    local_strategy_radius :int =50 ,
    positive_quality_threshold :float =0.2 ,
    ratio_dedup_delta :float =0.05 ,
    precomputed_reps_path :Optional [str ]=None ,
    quality_weight_alpha :float =5.0 ,
    quality_weight_min :float =0.1 ,
    enable_ratio_dedup :bool =False ,
    use_contrastive_loss :bool =True ,
    preferred_sample_weight :float =3.0 ,
    non_preferred_sample_weight :float =0.1 ,
    ):
        """ Args: csv_path: MMPDataCSVDocumentationPath encoder_model: TrainingIt is. .. CGIPEncoder（If. .. UseCalculateEssaysavailableNone） target_property: Target Optimisation Properties (mw, qed, saWait.) property_list: andTrainingIt is. .. ；If providedPressThe. .. transform: Image Change negative_sample_size: Every one. Negative sampleCount properties_to_preserve: Columns correlation_file: RelevanceOutcomeDocumentationPath deviation_threshold: Threshold，For local_strategy_radius: Policy1andPolicy2It is. .. SelectionRadius positive_quality_threshold: Positive sampleThreshold，ValueIt is. .. As a. .. Positive sample ratio_dedup_delta: Start，Willratioavailable±deltaIt is. .. ，Reservation onlyIt is. .. """

        self .data =pd .read_csv (csv_path )
        print (f" Data: {len(self. data): ,} CLI")
        self .property_label_column =property_label_column 
        self .property_labels :Optional [pd .Series ]=None 
        self .property_vocab :List [str ]=[]
        self .property_to_idx :Dict [str ,int ]={}

        property_list_clean :List [str ]=[]
        if property_list :
            if isinstance (property_list ,str ):
                property_list_clean =[p .strip ()for p in property_list .split (', ')if p .strip ()]
            else :
                property_list_clean =[str (p ).strip ()for p in property_list if str (p ).strip ()]

        if self .property_label_column :
            if self .property_label_column not in self .data .columns :
                raise ValueError (f"availableDataMediumFoundColumns: {self. property_label_column}")
            self .property_labels =self .data [self .property_label_column ].astype (str ).fillna ('')

            if property_list_clean :
                before_filter =len (self .data )
                self .data =self .data [self .data [self .property_label_column ].isin (property_list_clean )].reset_index (drop =True )
                self .property_labels =self .data [self .property_label_column ].astype (str ).fillna ('')
                after_filter =len (self .data )
                if after_filter ==0 :
                    raise ValueError (
                    f"Columns {self. property_label_column} MediumOrganisation property_list MediumIt is. .. : {', '. join(property_list_clean)}"
                    )
                if after_filter <before_filter :
                    print (f" According to property_list Filter Sample: {before_filter: ,} -> {after_filter: ,}")

            self .property_vocab =property_list_clean if property_list_clean else sorted (self .property_labels .unique ().tolist ())
            self .property_to_idx ={name :idx for idx ,name in enumerate (self .property_vocab )}

            if filter_by_property_label :
                before =len (self .data )
                self .data =self .data [
                self .data [self .property_label_column ].astype (str )==str (target_property )
                ].reset_index (drop =True )
                self .property_labels =self .data [self .property_label_column ].astype (str ).fillna ('')
                after =len (self .data )
                if after ==0 :
                    raise ValueError (
                    f"Columns {self. property_label_column} MediumNo Objective {target_property} "
                    )
                print (f" According to {self. property_label_column}=={target_property} Filter: {before: ,} -> {after: ,} CLI")
                self .property_vocab =[target_property ]
                self .property_to_idx ={target_property :0 }
            else :
                print (f" DetectedColumns {self. property_label_column}，WillPressSelection（Total {len(self. property_vocab)} Classes）")

                # Set Properties
        self .encoder =encoder_model 
        self .target_property =target_property 
        self .use_contrastive_loss =bool (use_contrastive_loss )
        self .negative_sample_size =negative_sample_size if self .use_contrastive_loss and negative_sample_size >0 else 0 
        self .deviation_threshold =deviation_threshold 
        self .batch_size =batch_size 
        self .local_strategy_radius =local_strategy_radius 
        self .positive_quality_threshold =positive_quality_threshold 
        self .ratio_dedup_delta =ratio_dedup_delta 
        self .precomputed_reps_path =precomputed_reps_path 
        self .all_properties =self ._detect_available_properties ()
        if self .property_vocab :
            missing_cfg =[p for p in self .property_vocab if p not in PROPERTY_CONFIG ]
            missing_cols =[p for p in self .property_vocab if p not in self .all_properties ]
            if missing_cfg :
                raise ValueError (f"available PROPERTY_CONFIG MediumFound: {', '. join(missing_cfg)}")
            if missing_cols :
                raise ValueError (f"DataMediumFoundColumns: {', '. join(missing_cols)}")
        else :
            if self .target_property not in PROPERTY_CONFIG :
                raise ValueError (f"availablePROPERTY_CONFIGMediumFoundObjective: {self. target_property}")
            if self .target_property not in self .all_properties :
                raise ValueError (f"DataMediumFoundObjectiveColumns: {self. target_property}")
                # Single Properties vocab
            self .property_vocab =[self .target_property ]
            self .property_to_idx ={self .target_property :0 }

        if isinstance (properties_to_preserve ,str ):
            properties_to_preserve =[prop .strip ()for prop in properties_to_preserve .split (', ')if prop .strip ()]

        if properties_to_preserve is None or len (properties_to_preserve )==0 :
            if property_list_clean :
                self .properties_to_preserve =list (dict .fromkeys (property_list_clean ))
            elif self .property_vocab :
                self .properties_to_preserve =list (dict .fromkeys (self .property_vocab ))
            else :
                self .properties_to_preserve =[prop for prop in self .all_properties if prop !=self .target_property ]
        else :
            self .properties_to_preserve =list (dict .fromkeys (properties_to_preserve ))

        for prop in list (dict .fromkeys ([self .target_property ]+self .properties_to_preserve )):
            if prop not in PROPERTY_CONFIG :
                raise ValueError (f"availablePROPERTY_CONFIGMediumFound: {prop}")
            if prop not in self .all_properties :
                raise ValueError (f"DataMediumFoundColumns: {prop}")

        self .properties_to_preserve =[
        prop for prop in self .properties_to_preserve 
        if prop in self .all_properties 
        ]

        self .property_columns =self ._resolve_all_property_columns ()
        # Success/FailedTag（IfDataMedium）
        self .success_mask :Optional [np .ndarray ]=None 
        self .failure_indices_by_start :Optional [Dict [str ,List [int ]]]=None 
        self .failure_indices_all :Optional [List [int ]]=None 
        if 'is_success'in self .data .columns :
            self .success_mask =self .data ['is_success'].astype (bool ).to_numpy ()

        self .quality_weight_alpha =max (0.0 ,float (quality_weight_alpha ))
        self .quality_weight_min =max (0.0 ,float (quality_weight_min ))
        self .enable_ratio_dedup =enable_ratio_dedup 
        self .preferred_sample_weight =max (float (preferred_sample_weight ),0.0 )
        self .non_preferred_sample_weight =max (float (non_preferred_sample_weight ),0.0 )
        if self .non_preferred_sample_weight ==0.0 :
            print (" ⚠️ non_preferred_sample_weight Set As0，It could cause the gradient to disappear. ，Use")

            # DefaultImage Change, anddiffusionConsistency in training
        if transform is None :
            self .transform =transforms .Compose ([
            transforms .Resize ((224 ,224 )),
            transforms .ToTensor (),
            transforms .Lambda (scale_to_range )
            ])
        else :
            self .transform =transform 

            # Loading nature-relevance information
        print ("Loadandused inCalculateIt is. .. RelevanceInformation、Relevance matrix. .. ")
        self .property_correlations =self ._load_property_correlations (correlation_file )
        self .correlation_matrix =self ._build_correlation_matrix ()

        # Initialise Cache（ForClassesCalculatePath; ReadCalculateCache）
        self .representation_cache ={}

        # DatasetMediumLoadorCalculateEssays, Changed fromtrain. pyProcess
        self .use_precomputed_reps =False 

        # CalculateValueForNegative sampleSelection
        print ("Here we go. CalculateandCount. .. ")
        """ Calculateused in，OtherIt is. .. （We've subtracted the correlation. ） scores[0] = 0.1 # It is perfect. scores[1] = 0.5 # It is a better solution. scores[2] = 0.8 # Poor decoupling. One. CountCountFor： - Screening of high-quality training samples（CountIt is. .. ） - Negative sampleSelectionPolicy（SelectionCountIt is. .. As a. .. Negative sample） """
        self ._precompute_ratios ()

        # InitializationCandidatesCache（ForSpeed up. Negative sampleSelection）
        self .candidate_pool_cache ={
        'center_idx':-5000 ,# CandidatesMediumIndex, ValueMake sure.
        'pool_indices':None ,# CandidatesIndexCount
        'pool_scores':None ,# CandidatesCount
        'pool_ratios':None ,# Candidates
        'pool_start':-1 ,# CandidatesStartIndex
        'pool_end':-1 # CandidatesIndex
        }
        self .pool_radius =5000 # CandidatesRadius（Up and down. 5000A sample. ）
        self .pool_update_threshold =1000 # idxMediumMore thanValueCandidates
        print (f" UseCandidatesPolicy: Radius={self. pool_radius}, Threshold={self. pool_update_threshold}")

    def _detect_available_properties (self )->List [str ]:
        """According toDataColumnsUse it. """
        available =[]
        for prop in PROPERTY_CONFIG .keys ():
            start_col =self ._resolve_column_name (prop ,'start',raise_error =False )
            final_col =self ._resolve_column_name (prop ,'final',raise_error =False )
            if start_col and final_col :
                available .append (prop )
        if not available :
            raise ValueError ("availableDataMediumFoundColumns，InspectionCSVColumns")
        return available 

    def _resolve_all_property_columns (self )->Dict [str ,Tuple [str ,str ]]:
        """start/finalColumns，Compatibility_predand"""
        mapping :Dict [str ,Tuple [str ,str ]]={}
        for prop in self .all_properties :
            start_col =self ._resolve_column_name (prop ,'start')
            final_col =self ._resolve_column_name (prop ,'final')
            mapping [prop ]=(start_col ,final_col )
        return mapping 

    def _resolve_column_name (self ,prop :str ,prefix :str ,raise_error :bool =True )->Optional [str ]:
        candidates =[
        f"{prefix}_{prop}",
        f"{prefix}_{prop}_pred",
        f"{prefix}_{prop. lower()}",
        f"{prefix}_{prop. lower()}_pred",
        f"{prefix}_{prop. capitalize()}",
        f"{prefix}_{prop. capitalize()}_pred",
        ]
        for candidate in candidates :
            if candidate in self .data .columns :
                return candidate 
        if raise_error :
            raise KeyError (f"availableDataMediumFoundColumns: {prop} (prefix={prefix})")
        return None 

    def _get_property_values (self ,prop :str )->Tuple [np .ndarray ,np .ndarray ]:
        start_col ,final_col =self .property_columns [prop ]
        start_series =pd .to_numeric (self .data [start_col ],errors ='coerce')
        final_series =pd .to_numeric (self .data [final_col ],errors ='coerce')
        start_values =start_series .values .astype (np .float32 )
        final_values =final_series .values .astype (np .float32 )
        start_values =np .nan_to_num (start_values ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        final_values =np .nan_to_num (final_values ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        return start_values ,final_values 



    def _build_correlation_matrix (self )->Dict [str ,float ]:
        """BuildIt is. .. Relevance matrix"""
        correlation_matrix ={}

        # DefaultRelevance0
        all_properties =set (self .all_properties )
        for key in self .property_correlations .keys ():
            if '-'in key :
                prop1 ,prop2 =key .split ('-',1 )
                all_properties .add (prop1 )
                all_properties .add (prop2 )

        for prop1 in all_properties :
            for prop2 in all_properties :
                if prop1 !=prop2 :
                    key =f"{prop1}-{prop2}"
                    correlation_matrix [key ]=0.0 

                    # RelevanceValue
        for key ,value in self .property_correlations .items ():
            correlation_matrix [key ]=value 

        return correlation_matrix 


    def _calculate_decoupling_scores_vectorized (self )->np .ndarray :
        """PressClasses/ReturnsCalculateCount"""
        num_samples =len (self .data )
        if num_samples ==0 :
            return np .array ([],dtype =np .float32 )

        if self .property_labels is None :
            total_penalty =np .zeros (num_samples ,dtype =np .float32 )
            properties =list (dict .fromkeys ([self .target_property ,*self .properties_to_preserve ]))
            total_properties =max (len (properties ),1 )

            for prop in properties :
                start_vals ,final_vals =self ._get_property_values (prop )
                penalty =_binary_penalty_array (
                start_vals ,
                final_vals ,
                prop ,
                is_target =(prop ==self .target_property ),
                )
                penalty =np .nan_to_num (penalty ,nan =1.0 ,posinf =1.0 ,neginf =1.0 )
                total_penalty +=penalty 

            scores =total_penalty /total_properties 
            scores =np .nan_to_num (scores ,nan =1.0 ,posinf =1.0 ,neginf =1.0 )
            return scores .astype (np .float32 )

        scores =np .zeros (num_samples ,dtype =np .float32 )
        for prop in self .property_vocab :
            mask =(self .property_labels ==prop )
            if not mask .any ():
                continue 
            mask_idx =mask .to_numpy ()
            total_penalty =np .zeros (mask .sum (),dtype =np .float32 )

            start_vals ,final_vals =self ._get_property_values (prop )
            penalty_target =_binary_penalty_array (
            start_vals [mask_idx ],
            final_vals [mask_idx ],
            prop ,
            is_target =True ,
            )
            total_penalty +=np .nan_to_num (penalty_target ,nan =1.0 ,posinf =1.0 ,neginf =1.0 )

            preserve_props =[p for p in self .properties_to_preserve if p !=prop ]
            total_properties =max (1 +len (preserve_props ),1 )
            for pres in preserve_props :
                s_vals ,f_vals =self ._get_property_values (pres )
                penalty_pres =_binary_penalty_array (
                s_vals [mask_idx ],
                f_vals [mask_idx ],
                pres ,
                is_target =False ,
                )
                total_penalty +=np .nan_to_num (penalty_pres ,nan =1.0 ,posinf =1.0 ,neginf =1.0 )

            scores [mask_idx ]=total_penalty /total_properties 

        scores =np .nan_to_num (scores ,nan =1.0 ,posinf =1.0 ,neginf =1.0 )
        return scores .astype (np .float32 )


    def _load_property_correlations (self ,correlation_file :Optional [str ])->Dict :
        """Loading nature-relevance information"""

        correlations ={}

        if not correlation_file :
            print (" ℹ️ RelevanceDocumentation，DefaultRelevanceavailable0")
            return correlations 

        if not os .path .exists (correlation_file ):
            print (f" ⚠️ RelevanceDocumentationavailable: {correlation_file}，WillUseDefault0Relevance")
            return correlations 

        try :
            with open (correlation_file ,'r')as f :
                correlation_data =json .load (f )
        except (OSError ,json .JSONDecodeError )as exc :
            print (f" ⚠️ ReadRelevanceDocumentation({correlation_file}): {exc}，WillUseDefault0Relevance")
            return correlations 

        if 'pearson_matrix'in correlation_data :
            pearson_matrix =correlation_data ['pearson_matrix']or {}
            for prop1 ,row in pearson_matrix .items ():
                if not isinstance (row ,dict ):
                    continue 
                for prop2 ,value in row .items ():
                    if prop1 ==prop2 :
                        continue 
                    try :
                        corr_value =float (value )
                    except (TypeError ,ValueError ):
                        continue 
                    correlations [f"{prop1}-{prop2}"]=corr_value 
        else :
            print (" ⚠️ availableRelevanceDocumentationMediumFoundUse it. Data (pearson_matrix)，WillUseDefault0Relevance")

            # Make sure.
        for key ,value in list (correlations .items ()):
            prop1 ,prop2 =key .split ('-')
            symmetric_key =f"{prop2}-{prop1}"
            if symmetric_key not in correlations :
                correlations [symmetric_key ]=value 

        loaded_pairs =len (correlations )//2 
        if loaded_pairs >0 :
            print (f" ✓ SuccessLoad {loaded_pairs} One. Relevance")

            target_correlations =[]
            for key ,value in correlations .items ():
                if key .startswith (f"{self. target_property}-"):
                    other_prop =key .split ('-')[1 ]
                    target_correlations .append (f"{other_prop}(r={value: .3f})")

            if target_correlations :
                print (f" 📊 Destination Properties {self. target_property} Relevance: {', '. join(target_correlations)}")
        else :
            print (" ℹ️ FromRelevanceDocumentationMediumPresent. ValidityData，WillUseDefault0Relevance")

        return correlations 


    def _precompute_ratios (self ):
        """Calculate、CountWeight"""
        print (f" 🔄 CalculateDestination Properties {self. target_property}. .. ")
        self .target_changes =self ._compute_target_change_metric ()
        self .target_changes =np .nan_to_num (self .target_changes ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        # Compatibility, ratiosSave
        self .ratios =self .target_changes .copy ()

        print (f" 🔄 Calculate {len(self. data): ,} A sample. Count. .. ")

        self .decoupling_scores =self ._calculate_decoupling_scores_vectorized ()
        self .decoupling_scores =np .nan_to_num (self .decoupling_scores ,nan =1.0 ,posinf =1.0 ,neginf =1.0 )
        self .preference_mask =self ._build_preference_mask ()
        preferred_count =int (self .preference_mask .sum ())
        print (
        f" ⭐ Preferred Sample: {preferred_count: ,}/{len(self. data): ,} "
        f"({preferred_count/len(self. data)*100: .1f}%)"
        )

        # Statistical high-quality samples, For
        high_quality_mask =self .decoupling_scores <self .positive_quality_threshold 
        high_quality_count =int (high_quality_mask .sum ())
        print (
        f" ✅ Quality sample: {high_quality_count: ,}/{len(self. data): ,} "
        f"({high_quality_count/len(self. data)*100: .1f}%)，TrainingWillUse"
        )

        # UseSuccessAs a Training（Like）, Yes
        if self .success_mask is not None :
            self .valid_indices =np .where (self .success_mask )[0 ]
        else :
            self .valid_indices =np .arange (len (self .data ))
        self .sample_weights =self ._compute_quality_weights ()

        # Reservation onlyClasses（Like BBBP=0, hERG=1, Mutagenicity=1）
        self ._filter_by_preferred_start_values ()

        if self .enable_ratio_dedup and self .ratio_dedup_delta >0 :
            self ._deduplicate_by_ratio_window ()
        else :
            print (" 🔁 RatioWindowUse it. ，ForTraining")

        self ._build_failure_pools ()

    def _compute_target_change_metric (self )->np .ndarray :
        """According toDestination PropertiesClassesCalculate；PressSelection。"""
        if self .property_labels is None :
            start_vals ,final_vals =self ._get_property_values (self .target_property )
            change =final_vals -start_vals 
            change =np .nan_to_num (change ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
            return change .astype (np .float32 )

        changes =np .zeros (len (self .data ),dtype =np .float32 )
        for prop in self .property_vocab :
            mask =(self .property_labels ==prop )
            if not mask .any ():
                continue 
            mask_idx =mask .to_numpy ()
            start_vals ,final_vals =self ._get_property_values (prop )
            delta =final_vals -start_vals 
            delta =np .nan_to_num (delta ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
            changes [mask_idx ]=delta [mask_idx ]
        return changes .astype (np .float32 )

    def _build_preference_mask (self )->np .ndarray :
        """According toDestination PropertiesIt is. .. ConditionGenerate"""
        if self .property_labels is None :
            start_vals ,final_vals =self ._get_property_values (self .target_property )
            config =PROPERTY_CONFIG [self .target_property ]

            if config .get ('type')=='classification':
                transition =config .get ('preferred_transition',{})
                start_target =transition .get ('start')
                final_target =transition .get ('final')
                preferred =np .ones_like (start_vals ,dtype =bool )
                if start_target is not None :
                    preferred &=np .isclose (start_vals ,start_target ,atol =0.5 )
                if final_target is not None :
                    preferred &=np .isclose (final_vals ,final_target ,atol =0.5 )
            else :
                required_delta =float (config .get ('preferred_delta',0.5 ))
                preferred =(final_vals -start_vals )>=required_delta 

            return preferred 

        preferred =np .zeros (len (self .data ),dtype =bool )
        for prop in self .property_vocab :
            mask =(self .property_labels ==prop )
            if not mask .any ():
                continue 
            mask_idx =mask .to_numpy ()
            start_vals ,final_vals =self ._get_property_values (prop )
            start_vals =start_vals [mask_idx ]
            final_vals =final_vals [mask_idx ]
            config =PROPERTY_CONFIG [prop ]
            if config .get ('type')=='classification':
                transition =config .get ('preferred_transition',{})
                start_target =transition .get ('start')
                final_target =transition .get ('final')
                local_pref =np .ones_like (start_vals ,dtype =bool )
                if start_target is not None :
                    local_pref &=np .isclose (start_vals ,start_target ,atol =0.5 )
                if final_target is not None :
                    local_pref &=np .isclose (final_vals ,final_target ,atol =0.5 )
            else :
                required_delta =float (config .get ('preferred_delta',0.5 ))
                local_pref =(final_vals -start_vals )>=required_delta 
            preferred [mask_idx ]=local_pref 
        return preferred 


    def _deduplicate_by_ratio_window (self ):
        """startavailableratioScopeIt is. .. ，Keep OnlyIt is. .. """

        if len (self .valid_indices )==0 :
            return 

            # ratioWindow: ±ratio_dedup_delta, Total span approximately0.1
        ratio_window =max (self .ratio_dedup_delta *2 ,0.0 )
        if ratio_window <=0 :
            return 

        start_values =self .data ['start'].values 
        candidate_groups =defaultdict (list )

        for idx in self .valid_indices :
            start_smiles =start_values [idx ]
            candidate_groups [start_smiles ].append (idx )

        filtered_indices =[]

        for start_smiles ,indices in candidate_groups .items ():
            if len (indices )==1 :
                filtered_indices .append (indices [0 ])
                continue 

                # PressPresent. ,
            indices_sorted =sorted (indices ,key =lambda i :self .decoupling_scores [i ])
            kept_ratios :List [float ]=[]

            for idx in indices_sorted :
                ratio_value =float (self .ratios [idx ])

                # IfWindow,
                conflict =False 
                for kept_ratio in kept_ratios :
                    if abs (ratio_value -kept_ratio )<=ratio_window :
                        conflict =True 
                        break 

                if conflict :
                    continue 

                filtered_indices .append (idx )
                kept_ratios .append (ratio_value )

        filtered_indices =np .array (sorted (filtered_indices ))

        removed_count =len (self .valid_indices )-len (filtered_indices )
        remaining_count =len (filtered_indices )
        if removed_count >0 :
            print (
            f" 🔁 RatioWindow: Removed. {removed_count: ,} Article，Keep Only ±{self. ratio_dedup_delta: .2f} Best sample inside. "
            f"；Remaining {remaining_count: ,} Article"
            )
        else :
            print (f" ✅ RatioWindow: No duplicate sample to remove，Total {remaining_count: ,} Article")

        self .valid_indices =filtered_indices 

    def _build_failure_pools (self )->None :
        """BuildStartIt is. .. Failed，ForNegative sampleSample。"""
        if self .success_mask is None :
            self .failure_indices_by_start =None 
            self .failure_indices_all =None 
            return 

        failure_indices =np .where (~self .success_mask )[0 ]
        if failure_indices .size ==0 :
            self .failure_indices_by_start ={}
            self .failure_indices_all =[]
            print (" ℹ️ DataMediumOrganisationFailed，Negative sampleWillavailableCandidatesPolicy")
            return 

        start_series =self .data ['start'].fillna ('').astype (str ).to_numpy ()
        failure_map :Dict [str ,List [int ]]=defaultdict (list )
        for idx in failure_indices :
            failure_map [start_series [idx ]].append (int (idx ))

        self .failure_indices_by_start =failure_map 
        self .failure_indices_all =failure_indices .astype (int ).tolist ()
        print (
        f" 📉 Failed sample pool: {len(self. failure_indices_all): ,} Article，"
        f"{len(self. failure_indices_by_start)} It is unique. start"
        )

    def _filter_by_preferred_start_values (self )->None :
        """Reservation onlyClassesStartValueIt is. .. ；Resume Properties Keep All。"""
        if len (self .valid_indices )==0 :
            return 

            # CacheClassesStartValue
        start_cache :Dict [str ,np .ndarray ]={}
        for prop in self .property_vocab :
            cfg =PROPERTY_CONFIG .get (prop ,{})
            if cfg .get ("type")=="classification":
                start_vals ,_ =self ._get_property_values (prop )
                start_cache [prop ]=start_vals 

        labels =self .property_labels .to_numpy ()if self .property_labels is not None else None 
        keep_mask =np .zeros (len (self .data ),dtype =bool )

        if labels is None :
            props_iter =[(self .target_property ,np .ones (len (self .data ),dtype =bool ))]
        else :
            props_iter =[(prop ,labels ==prop )for prop in self .property_vocab ]

        for prop ,base_mask in props_iter :
            cfg =PROPERTY_CONFIG .get (prop ,{})
            prop_type =cfg .get ("type","regression")
            local_mask =base_mask .copy ()
            if prop_type =="classification":
                transition =cfg .get ("preferred_transition",{})
                start_target =transition .get ("start")
                if start_target is not None and prop in start_cache :
                    start_vals =start_cache [prop ]
                    tol =float (cfg .get ("classification_tolerance",CLASSIFICATION_TOLERANCE ))
                    local_mask =local_mask &np .isclose (start_vals ,float (start_target ),atol =tol )
            keep_mask |=local_mask 

        filtered_indices =np .array ([idx for idx in self .valid_indices if keep_mask [idx ]])
        removed =len (self .valid_indices )-len (filtered_indices )
        self .valid_indices =filtered_indices 
        if removed >0 :
            print (
            f" 🧹 PressStartFilter: Remove {removed: ,} ArticleIt is. .. Classes，Remaining {len(self. valid_indices): ,} Article"
            )


    def _compute_quality_weights (self )->np .ndarray :
        """According toCountCalculateTrainingWeight"""
        scores =self .decoupling_scores 
        if scores is None or len (scores )==0 :
            return np .array ([],dtype =np .float32 )

        if self .quality_weight_alpha <=0 :
            weights =np .ones_like (scores ,dtype =np .float32 )
        else :
            weights =np .exp (-self .quality_weight_alpha *scores ).astype (np .float32 )
            if self .quality_weight_min >0 :
                weights =np .maximum (weights ,self .quality_weight_min )

        if hasattr (self ,'preference_mask')and self .preference_mask is not None :
            preference_factor =np .where (
            self .preference_mask ,
            max (self .preferred_sample_weight ,0.0 ),
            max (self .non_preferred_sample_weight ,0.0 )
            ).astype (np .float32 )
            weights =weights *preference_factor 

        weights =np .nan_to_num (weights ,nan =1.0 ,posinf =1.0 ,neginf =1.0 )

        # Normalize to Average1, Maintain overall gradient scale stable
        mean_val =float (np .mean (weights ))if len (weights )>0 else 1.0 
        if mean_val <=0 :
            mean_val =1.0 
        weights =weights /mean_val 

        print (
        " ⚖️ Training sample weight statistics: "
        f"min={weights. min(): .3f}, max={weights. max(): .3f}, "
        f"mean={weights. mean(): .3f}"
        )

        return weights .astype (np .float32 )


    def _precompute_all_representations (self ):
        """CalculateIt is. .. Essays，AvoidProcessCUDAProblem"""
        unique_smiles =set ()

        # CollectionOnlySMILES
        for _ ,row in self .data .iterrows ():
            unique_smiles .add (row ['start'])
            unique_smiles .add (row ['final'])

        unique_smiles =list (unique_smiles )
        print (f" 🧮 Calculate {len(unique_smiles): ,} One. OnlyIt is. .. Essays. .. ")

        # BatchCalculateEssays
        batch_size =32 
        for i in range (0 ,len (unique_smiles ),batch_size ):
            batch_smiles =unique_smiles [i :i +batch_size ]

            for smiles in batch_smiles :
                if smiles not in self .representation_cache :
                    try :
                    # GenerateImage
                        img =self .mol_to_image (smiles )

                        # Convert totensor
                        if self .transform :
                            img_tensor =self .transform (img )

                            # AddbatchDimensions
                        img_tensor =img_tensor .unsqueeze (0 )

                        # UseEncoderGetEssays
                        with torch .no_grad ():
                            device =next (self .encoder .parameters ()).device 
                            img_tensor =img_tensor .to (device )
                            rep =self .encoder (img_tensor ).squeeze ()# (512,)

                            # Standardization（anddiffusionconsistent Z-scoreStandardization）
                            if rep .numel ()>1 :
                                rep =normalize_representation (rep .unsqueeze (0 )).squeeze (0 )

                        rep_cpu =rep .cpu ()

                        # Save to Cache
                        self .representation_cache [smiles ]=rep_cpu .clone ()

                    except Exception as e :
                        print (f"This function is EXPERIMENTAL. : CalculateSMILES '{smiles}' Failed: {e}")
                        # StorageAs a fallback
                        self .representation_cache [smiles ]=torch .zeros (512 )

                        # Show Progress
            if (i +batch_size )%(batch_size *10 )==0 or i +batch_size >=len (unique_smiles ):
                progress =min (i +batch_size ,len (unique_smiles ))
                print (f" Progress: {progress: ,}/{len(unique_smiles): ,} ({progress/len(unique_smiles)*100: .1f}%)")

        print (f" ✅ Calculate，Cache. {len(self. representation_cache): ,} A molecule. Essays")

        # DatasetI. C. Cache
    pass 


    def mol_to_image (self ,smiles :str )->Image .Image :
        """WillSMILESConvert toImage"""
        mol =Chem .MolFromSmiles (smiles )
        if mol is None :
            raise ValueError (f"Invalid SMILES: {smiles}")

            # GenerateImage
        img =Draw .MolToImage (mol ,size =(224 ,224 ))
        return img 


    def _select_negatives_candidate_pool (self ,idx :int )->List [int ]:
        """SelectionNegative sampleIndex：UseCandidatesPolicy Fromused inidxIt is. .. pool_radiusScopeSelectionNegative sample， AvoidavailableDataCLI，Raise。 """
        if not self .use_contrastive_loss or self .negative_sample_size <=0 :
            return []

        current_ratio =self .ratios [idx ]
        current_score =self .decoupling_scores [idx ]
        negative_indices =[]

        # InspectionYesCandidates
        if (abs (idx -self .candidate_pool_cache ['center_idx'])>self .pool_update_threshold or 
        self .candidate_pool_cache ['pool_indices']is None ):

        # CalculateCandidatesScope
            pool_start =max (0 ,idx -self .pool_radius )
            pool_end =min (len (self .data ),idx +self .pool_radius +1 )

            # CandidatesCache
            self .candidate_pool_cache ['center_idx']=idx 
            self .candidate_pool_cache ['pool_start']=pool_start 
            self .candidate_pool_cache ['pool_end']=pool_end 

            # GetCandidatesIndex（）
            pool_indices =np .arange (pool_start ,pool_end )
            pool_indices =pool_indices [pool_indices !=idx ]

            # CacheCandidates
            self .candidate_pool_cache ['pool_indices']=pool_indices 
            self .candidate_pool_cache ['pool_scores']=self .decoupling_scores [pool_indices ]
            self .candidate_pool_cache ['pool_ratios']=self .ratios [pool_indices ]

            # UseCacheCandidates
        available_indices =self .candidate_pool_cache ['pool_indices'].copy ()
        pool_scores =self .candidate_pool_cache ['pool_scores']
        pool_ratios =self .candidate_pool_cache ['pool_ratios']

        # Make sure. idxCandidatesMedium（Processing）
        if idx >=self .candidate_pool_cache ['pool_start']and idx <self .candidate_pool_cache ['pool_end']:
        # CalculateidxpoolMedium
            relative_idx =idx -self .candidate_pool_cache ['pool_start']
            if relative_idx <len (available_indices )and available_indices [relative_idx ]==idx :
            # Createidx
                mask =np .ones (len (available_indices ),dtype =bool )
                mask [relative_idx ]=False 
                available_indices =available_indices [mask ]
                pool_scores =pool_scores [mask ]
                pool_ratios =pool_ratios [mask ]

        if len (available_indices )==0 :
            return []

            # Policy1: Selection（Jim. 70%）- ScopeSelection
        strategy1_count =int (self .negative_sample_size *0.7 )

        if strategy1_count >0 :
        # CalculatePolicyScope
            local_start =max (0 ,idx -self .local_strategy_radius )
            local_end =min (len (self .data ),idx +self .local_strategy_radius +1 )

            # GetScopeCandidates（）
            local_indices =np .arange (local_start ,local_end )
            local_indices =local_indices [local_indices !=idx ]

            # FilterCandidates
            local_in_pool =np .intersect1d (local_indices ,available_indices )

            if len (local_in_pool )>0 :
            # GetCount
                local_mask =np .isin (available_indices ,local_in_pool )
                local_scores =pool_scores [local_mask ]

                # UseThreshold（andPositive sampleconsistent ）
                quality_threshold =self .positive_quality_threshold 

                # SelectionThreshold（Low quality sample）
                poor_quality_mask =local_scores >quality_threshold 
                poor_quality_indices =local_in_pool [poor_quality_mask ]

                if len (poor_quality_indices )>0 :
                    selected_count =min (strategy1_count ,len (poor_quality_indices ))
                    selected =np .random .choice (
                    poor_quality_indices ,
                    size =selected_count ,
                    replace =False 
                    )
                    negative_indices .extend (selected )

                    # Policy2: Selection（Jim. Remaining）- ScopeSelection
        remaining_needed =self .negative_sample_size -len (negative_indices )
        strategy2_count =min (remaining_needed //2 ,remaining_needed )

        if strategy2_count >0 :
        # CalculatePolicyScope（Use it. strategy1Scope）
            local_start =max (0 ,idx -self .local_strategy_radius )
            local_end =min (len (self .data ),idx +self .local_strategy_radius +1 )

            # GetScopeCandidates（and）
            local_indices =np .arange (local_start ,local_end )
            local_indices =local_indices [local_indices !=idx ]

            # FilterCandidatesMedium
            local_in_pool =np .intersect1d (local_indices ,available_indices )
            selected_mask =np .isin (local_in_pool ,negative_indices )
            remaining_local =local_in_pool [~selected_mask ]

            if len (remaining_local )>0 :
            # Get
                local_pool_mask =np .isin (available_indices ,remaining_local )
                remaining_local_ratios =pool_ratios [local_pool_mask ]

                # 
                opposite_direction_mask =(
                (current_ratio >1 and remaining_local_ratios <1 )|
                (current_ratio <=1 and remaining_local_ratios >1 )
                )
                opposite_indices =remaining_local [opposite_direction_mask ]

                if len (opposite_indices )>0 :
                    selected_count =min (strategy2_count ,len (opposite_indices ))
                    selected =np .random .choice (
                    opposite_indices ,
                    size =selected_count ,
                    replace =False 
                    )
                    negative_indices .extend (selected )

                    # Policy3: Policy - one CandidatesMediumSelectionRemaining
        if len (negative_indices )<self .negative_sample_size :
            selected_mask =np .isin (available_indices ,negative_indices )
            remaining_indices =available_indices [~selected_mask ]

            if len (remaining_indices )>0 :
                needed =self .negative_sample_size -len (negative_indices )
                selected_count =min (needed ,len (remaining_indices ))
                selected =np .random .choice (
                remaining_indices ,
                size =selected_count ,
                replace =False 
                )
                negative_indices .extend (selected )

        return negative_indices [:self .negative_sample_size ]

    def _select_negatives_from_failure_pool (self ,idx :int )->List [int ]:
        if self .failure_indices_by_start is None or self .negative_sample_size <=0 :
            return []

        start_smiles =self .data .iloc [idx ]['start']
        same_start_failures =self .failure_indices_by_start .get (start_smiles ,[])
        negatives :List [int ]=[]

        if same_start_failures :
            if len (same_start_failures )>=self .negative_sample_size :
                negatives .extend (
                np .random .choice (
                same_start_failures ,
                size =self .negative_sample_size ,
                replace =False 
                ).tolist ()
                )
            else :
                negatives .extend (same_start_failures )

        if len (negatives )<self .negative_sample_size :
            needed =self .negative_sample_size -len (negatives )
            other_pool =[
            idx_val for idx_val in (self .failure_indices_all or [])
            if idx_val not in same_start_failures and idx_val !=idx 
            ]
            if other_pool :
                replace_flag =len (other_pool )<needed 
                additional =np .random .choice (
                other_pool ,
                size =min (needed ,len (other_pool )),
                replace =False 
                ).tolist ()
                negatives .extend (additional )
                needed =self .negative_sample_size -len (negatives )
                if needed >0 and replace_flag :
                    negatives .extend (np .random .choice (other_pool ,size =needed ,replace =True ).tolist ())

        return negatives [:self .negative_sample_size ]

    def _select_negative_samples (self ,idx :int )->List [int ]:
        if not self .use_contrastive_loss or self .negative_sample_size <=0 :
            return []

        negatives =self ._select_negatives_from_failure_pool (idx )

        if len (negatives )<self .negative_sample_size :
            fallback =self ._select_negatives_candidate_pool (idx )
            for candidate in fallback :
                if candidate ==idx or candidate in negatives :
                    continue 
                negatives .append (candidate )
                if len (negatives )>=self .negative_sample_size :
                    break 

        if len (negatives )<self .negative_sample_size :
            remaining_candidates =[i for i in range (len (self .data ))if i !=idx ]
            if remaining_candidates :
                extra =np .random .choice (
                remaining_candidates ,
                size =self .negative_sample_size -len (negatives ),
                replace =True 
                ).tolist ()
                negatives .extend (extra )

        return negatives [:self .negative_sample_size ]


    def __len__ (self ):
        return len (self .valid_indices )

    def __getitem__ (self ,idx ):
    # UseIndexGetDataMediumIndex
        actual_idx =self .valid_indices [idx ]
        row =self .data .iloc [actual_idx ]
        current_property =(
        self .property_labels .iloc [actual_idx ]
        if self .property_labels is not None else self .target_property 
        )
        property_id =self .property_to_idx .get (current_property ,0 )

        # GetandObjectiveSMILES
        start_smiles =row ['start']
        final_smiles =row ['final']

        # Calculate（CompatibilityClasses/Returns）
        ratio =self .target_changes [actual_idx ]
        preserve_props_sample =[p for p in self .properties_to_preserve if p !=current_property ]

        negative_smiles =[]
        if self .use_contrastive_loss and self .negative_sample_size >0 :
            neg_indices =self ._select_negative_samples (actual_idx )

            for neg_idx in neg_indices :
                try :
                    neg_row =self .data .iloc [neg_idx ]
                    neg_smiles =neg_row ['final']
                    negative_smiles .append (neg_smiles )
                except Exception as e :
                    print (f"This function is EXPERIMENTAL. : Negative sample {neg_idx} Process Failed: {e}")
                    negative_smiles .append (start_smiles )# UseAs a fallback

                    # CalculateOther（ForLoss）; Promise. consistent, 0
        other_properties ={}
        for prop in self .properties_to_preserve :
            if prop ==current_property :
                other_properties [f'{prop}_change_rate']=0.0 
                continue 
            try :
                start_col ,final_col =self .property_columns [prop ]
                start_prop =row [start_col ]
                final_prop =row [final_col ]
                if start_prop !=0 and np .isfinite (start_prop )and np .isfinite (final_prop ):
                    other_properties [f'{prop}_change_rate']=(final_prop -start_prop )/start_prop 
                else :
                    other_properties [f'{prop}_change_rate']=0.0 
            except Exception :
                other_properties [f'{prop}_change_rate']=0.0 

                # Objective/CountValue
        target_start_col ,_ =self .property_columns [current_property ]
        target_start_value =sanitize_property_value (row [target_start_col ])
        preserve_property_values ={}
        for prop in preserve_props_sample :
            try :
                preserve_property_values [prop ]=sanitize_property_value (row [self .property_columns [prop ][0 ]])
            except Exception :
                preserve_property_values [prop ]=0.0 

                # AddCount（UseIndex）
        decoupling_score =self .decoupling_scores [actual_idx ]

        sample ={
        'ratio':torch .tensor (ratio ,dtype =torch .float32 ),
        'other_properties':other_properties ,
        'decoupling_score':torch .tensor (decoupling_score ,dtype =torch .float32 ),
        'quality_weight':torch .tensor (
        self .sample_weights [actual_idx ]if len (self .sample_weights )>actual_idx else 1.0 ,
        dtype =torch .float32 
        ),
        'source_smiles':start_smiles ,
        'target_property_start':torch .tensor (target_start_value ,dtype =torch .float32 ),
        'preserve_properties':{
        prop :torch .tensor (value ,dtype =torch .float32 )
        for prop ,value in preserve_property_values .items ()
        },
        'property_id':torch .tensor (property_id ,dtype =torch .long ),
        'property_name':current_property ,
        }

        if self .use_contrastive_loss and self .negative_sample_size >0 :
            sample ['negative_smiles']=negative_smiles 

        return sample 



def collate_fn (batch ):
    """ProcessingCount - IfOrganisationCalculateEssays，Collapse Directly to Threshold - Yes，BackSMILESTrainingCalculate """

    ratios =torch .stack ([item ['ratio']for item in batch ])
    decoupling_scores =torch .stack ([item ['decoupling_score']for item in batch ])
    quality_weights =torch .stack ([item ['quality_weight']for item in batch ])
    has_negatives ='negative_smiles'in batch [0 ]
    target_starts =torch .stack ([item ['target_property_start']for item in batch ])
    property_ids =torch .stack ([item ['property_id']for item in batch ])
    property_names =[item ['property_name']for item in batch ]

    # ProcessingOther
    other_properties ={}
    if batch [0 ]['other_properties']:
        for key in batch [0 ]['other_properties'].keys ():
            other_properties [key ]=torch .tensor (
            [item ['other_properties'][key ]for item in batch ],
            dtype =torch .float32 
            )
    else :
        other_properties ={}

    preserve_properties ={}
    #, Use it. 0Fill, Avoiding key incoherence leading to stacking errors
    all_preserve_keys =set ()
    for item in batch :
        all_preserve_keys .update (item .get ('preserve_properties',{}).keys ())
    for prop in sorted (all_preserve_keys ):
        values =[]
        for item in batch :
            if 'preserve_properties'in item and prop in item ['preserve_properties']:
                values .append (item ['preserve_properties'][prop ])
            else :
                values .append (torch .tensor (0.0 ,dtype =torch .float32 ))
        preserve_properties [prop ]=torch .stack (values )

        # CalculateEssaysPath
    if 'source_rep'in batch [0 ]:
        source_rep =torch .stack ([item ['source_rep']for item in batch ],dim =0 )# [B, 512]
        batch_dict ={
        'ratio':ratios ,
        'other_properties':other_properties ,
        'decoupling_score':decoupling_scores ,
        'source_rep':source_rep ,
        'quality_weight':quality_weights ,
        'target_property_start':target_starts ,
        'preserve_properties':preserve_properties ,
        'property_id':property_ids ,
        'property_name':property_names ,
        }
        if 'negatives'in batch [0 ]:
            batch_dict ['negatives']=torch .stack ([item ['negatives']for item in batch ],dim =0 )
        return batch_dict 

        # Dynamic Mode: BackSMILESandOtherInformation
    batch_dict ={
    'ratio':ratios ,
    'other_properties':other_properties ,
    'decoupling_score':decoupling_scores ,
    'quality_weight':quality_weights ,
    'source_smiles':[item ['source_smiles']for item in batch ],
    'target_property_start':target_starts ,
    'preserve_properties':preserve_properties ,
    'property_id':property_ids ,
    'property_name':property_names ,
    }
    if has_negatives :
        batch_dict ['negative_smiles']=[item ['negative_smiles']for item in batch ]
    return batch_dict 


class PropertyGroupedBatchSampler (torch .utils .data .Sampler ):
    """Press target_property BatchSampler，Promise. Every one. batch Internal properties are identical。"""

    def __init__ (
    self ,
    valid_indices :np .ndarray ,
    property_labels :pd .Series ,
    batch_size :int ,
    shuffle :bool =True ,
    ):
        """ Args: valid_indices: ♪ Passing through ♪FilterIt is. .. Use it. Location Index（and __getitem__ 0-based Subscript） property_labels: DataIt is. .. Series（Wait. DataCLICount） """
        self .valid_indices =valid_indices 
        self .property_labels =property_labels 
        self .batch_size =batch_size 
        self .shuffle =shuffle 

        # Press"Location Index"（and __getitem__ ）
        self .grouped_indices :Dict [str ,List [int ]]={}
        label_array =property_labels .to_numpy ()
        for pos ,real_idx in enumerate (valid_indices ):
            prop =str (label_array [real_idx ])
            self .grouped_indices .setdefault (prop ,[]).append (int (pos ))

    def __iter__ (self ):
    # Every one. Batchcolumns
        per_prop_batches :Dict [str ,List [List [int ]]]={}
        for prop ,idx_list in self .grouped_indices .items ():
            if self .shuffle :
                random .shuffle (idx_list )
            batches =[
            idx_list [i :i +self .batch_size ]
            for i in range (0 ,len (idx_list ),self .batch_size )
            ]
            per_prop_batches [prop ]=batches 

            # Batch, To avoid the hunger of the long tail properties.
        props_order =list (per_prop_batches .keys ())
        batch_pointers ={p :0 for p in props_order }
        emitted =0 
        total_batches =sum (len (batches )for batches in per_prop_batches .values ())

        while emitted <total_batches :
            for prop in props_order :
                ptr =batch_pointers [prop ]
                batches =per_prop_batches [prop ]
                if ptr <len (batches ):
                    yield batches [ptr ]
                    batch_pointers [prop ]+=1 
                    emitted +=1 

    def __len__ (self ):
        return sum (
        (len (idxs )+self .batch_size -1 )//self .batch_size 
        for idxs in self .grouped_indices .values ()
        )


def worker_init_fn (worker_id ):
    """ProcessworkerInitializationCount Make sure. Every one. workerProcessIt is. .. CandidatesCache，AvoidProcess。 """
    # GetProcessdataset
    worker_info =torch .utils .data .get_worker_info ()
    if worker_info is not None :
        dataset =worker_info .dataset 
        # Every one. workerInitializationCandidatesCache
        dataset .candidate_pool_cache ={
        'center_idx':-10000 ,
        'pool_indices':None ,
        'pool_scores':None ,
        'pool_ratios':None ,
        'pool_start':-1 ,
        'pool_end':-1 
        }
        # Random FeedsMake sure. Every one. worker
        np .random .seed (worker_id )
        # Use.pt Cache, No additional initialization required


def create_dataloader (
csv_path :str ,
encoder_model =None ,
batch_size :int =32 ,
shuffle :bool =True ,
group_by_property :bool =False ,
num_workers :int =0 ,
property_list :Optional [str ]=None ,
precomputed_reps_path :Optional [str ]=None ,
**dataset_kwargs 
):
# Dataset Again. Cache, UseProcessSpeed up.

    dataset =MMPDataset (
    csv_path ,
    encoder_model ,
    batch_size =batch_size ,
    property_list =property_list ,
    precomputed_reps_path =precomputed_reps_path ,
    **dataset_kwargs 
    )

    # If UseencoderModelnum_workers > 0, CalculateEssaysAvoidProcessCUDAProblem
    if encoder_model is not None and num_workers >0 :
        print (f"⚠️ DetectedCUDAModelandProcess (num_workers={num_workers})")
        dataset ._precompute_all_representations ()
        print (" ✅ Calculate，UseProcess")

        # Processmethod
    multiprocessing_context =None 
    if num_workers >0 :
        import multiprocessing as mp 
        try :
        # Usespawnmethod, CUDAMore friendly.
            multiprocessing_context =mp .get_context ('spawn')
            print (f"✅ UsespawnProcessMethodology，workers: {num_workers}")
        except RuntimeError :
        # If spawnUse it. , Back tofork
            print (f"⚠️ spawnMethodologyUse it. ，UseforkMethodology (It could lead to. .. CUDAProblem), workers: {num_workers}")
            multiprocessing_context =None 
    else :
        print ("UseProcess (num_workers=0)")

    if group_by_property :
        if dataset .property_labels is None :
            raise ValueError ("group_by_property Needs to be provided property_label_column")
        sampler =PropertyGroupedBatchSampler (
        dataset .valid_indices ,
        dataset .property_labels ,
        batch_size =batch_size ,
        shuffle =shuffle ,
        )
        dataloader =DataLoader (
        dataset ,
        batch_sampler =sampler ,
        num_workers =num_workers ,
        collate_fn =collate_fn ,
        multiprocessing_context =multiprocessing_context ,# Use it. Process
        persistent_workers =num_workers >0 ,
        pin_memory =True ,
        prefetch_factor =2 if num_workers >0 else 4 ,# ProcessJim. Use it.
        drop_last =False ,# Batch
        worker_init_fn =worker_init_fn if num_workers >0 else None ,# ProcessInitializationworker
        )
    else :
        dataloader =DataLoader (
        dataset ,
        batch_size =batch_size ,
        shuffle =False ,# Useshuffle, CandidatesPolicy
        num_workers =num_workers ,
        collate_fn =collate_fn ,
        multiprocessing_context =multiprocessing_context ,# Use it. Process
        persistent_workers =num_workers >0 ,
        pin_memory =True ,
        prefetch_factor =2 if num_workers >0 else 4 ,# ProcessJim. Use it.
        drop_last =False ,# Batch
        worker_init_fn =worker_init_fn if num_workers >0 else None ,# ProcessInitializationworker
        )

    return dataloader 
