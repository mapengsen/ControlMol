import os 
import sys 
import argparse 
import torch 
import torch .nn as nn 
import torch .optim as optim 
from torch .utils .data import DataLoader 
import numpy as np 
from tqdm import tqdm 
import json 
from datetime import datetime 
import multiprocessing as mp 
import time 
from concurrent .futures import ProcessPoolExecutor 
from rdkit import Chem 
from rdkit .Chem import Draw 
from PIL import Image 
import torchvision .transforms as transforms 
from line_profiler import profile 
import csv 
from safetensors .torch import save_file ,load_file 

import torch 


# Processmethod, SolveCUDAProcessProblem
if __name__ =='__main__':
    try :
        mp .set_start_method ('spawn',force =True )
    except RuntimeError :
        print ("This function is EXPERIMENTAL. : Unable to setspawnMethodology，Present. ProcessCUDAProblem")

        # AddPath
sys .path .append (os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))

from train_optimization_MMP .dataset import create_dataloader 
from train_optimization_MMP .model import ConditionalRepAdjuster 
from train_optimization_MMP .loss import OptimizationLoss ,train_step 
from train_optimization_MMP .train_property_predictor import (
load_checkpoint as load_property_predictor_checkpoint ,
)
import pretrained_enc .models_pretrained_enc as models_pretrained_enc 


def scale_to_range (x ):
    """Will[0, 1]Zoom to[-1, 1]"""
    return x *2.0 -1.0 


def normalize_representation (rep ):
    """anddiffusionModelTrainingconsistentZ-scoreStandardization"""
    rep_std =torch .std (rep ,dim =1 ,keepdim =True )
    rep_mean =torch .mean (rep ,dim =1 ,keepdim =True )
    rep_std =torch .clamp (rep_std ,min =1e-8 )
    return (rep -rep_mean )/rep_std 


    # Process（Fromprecompute_representations. pyPort）
class GlobalProcessPool :
    _instance =None 
    _pool =None 
    _num_processes =None 

    def __new__ (cls ):
        if cls ._instance is None :
            cls ._instance =super ().__new__ (cls )
        return cls ._instance 

    def initialize (self ,num_processes :int ):
        if self ._pool is None or self ._num_processes !=num_processes :
            if self ._pool is not None :
                self ._pool .shutdown (wait =True )

            self ._num_processes =num_processes 
            self ._pool =ProcessPoolExecutor (
            max_workers =num_processes ,
            mp_context =mp .get_context ('spawn')# UsespawnAvoidCUDAProblem
            )
            print (f" InitializationEssaysCalculateProcess: {num_processes} One. Process")

    def get_pool (self ):
        return self ._pool 

    def shutdown (self ):
        if self ._pool is not None :
            self ._pool .shutdown (wait =True )
            self ._pool =None 
            self ._num_processes =None 


def mol_to_image (smiles :str )->Image .Image :
    """WillSMILESConvert toImage"""
    try :
        mol =Chem .MolFromSmiles (smiles )
        if mol is None :
            raise ValueError (f"Invalid SMILES: {smiles}")

            # GenerateImage
        img =Draw .MolToImage (mol ,size =(224 ,224 ))
        return img 
    except Exception as e :
    # If Failed, CreateOne. Image
        print (f"This function is EXPERIMENTAL. : SMILES '{smiles}' Failed: {e}")
        return Image .new ('RGB',(224 ,224 ),color ='white')


def parallel_batch_mol_to_images (smiles_batch :list )->tuple :
    """ProcessCount：BatchProcessingSMILESColumns"""
    images =[]
    valid_indices =[]
    failed_count =0 

    for i ,smiles in enumerate (smiles_batch ):
        mol =Chem .MolFromSmiles (smiles )
        img =Draw .MolToImage (mol ,size =(256 ,256 ),imageType ='png',)

        images .append (img )
        valid_indices .append (i )

    return images ,valid_indices ,failed_count 


class DynamicRepresentationCalculator :
    """EssaysCalculate，Process"""

    def __init__ (self ,encoder_model ,num_processes :int =8 ,enable_multiprocess :bool =True ):
        self .encoder =encoder_model 
        self .num_processes =num_processes 
        self .enable_multiprocess =enable_multiprocess 

        # Image Change, anddiffusionConsistency in training
        self .transform =transforms .Compose ([
        transforms .ToTensor (),
        transforms .Lambda (scale_to_range )
        ])

        # EssaysCache
        self .representation_cache ={}

        self .pool_manager =None 
        self .pool =None 
        if self .enable_multiprocess and num_processes >1 :
            self .pool_manager =GlobalProcessPool ()
            self .pool_manager .initialize (num_processes )
            self .pool =self .pool_manager .get_pool ()
            print (f" 🚀 EssaysCalculateInitialization: Process ({num_processes} Process)")
        else :
            print (f" 🔄 EssaysCalculateInitialization: Process")


    def get_batch_representations (self ,smiles_list :list )->torch .Tensor :
        """BatchGetEssays"""
        batch_representations =[]

        # InspectionCache
        uncached_smiles =[]
        uncached_indices =[]

        for i ,smiles in enumerate (smiles_list ):
            if smiles in self .representation_cache :
                batch_representations .append (self .representation_cache [smiles ])
            else :
                uncached_smiles .append (smiles )
                uncached_indices .append (i )
                batch_representations .append (None )

                # CalculateCacheEssays
        if uncached_smiles :
            print ('CalculateCacheIt is. .. Essays')
            new_representations =self ._compute_multiprocess_representations (uncached_smiles )

            # FillOutcomeandCache
            for i ,(smiles ,rep )in enumerate (zip (uncached_smiles ,new_representations )):
                batch_idx =uncached_indices [i ]
                batch_representations [batch_idx ]=rep 
                self .representation_cache [smiles ]=rep 

                # Convert totensor
        return torch .stack (batch_representations )


    def _compute_multiprocess_representations (self ,smiles_list :list )->list :
    # Direct UseClassesMediumInitializationProcess
        if self .pool is None :
            raise RuntimeError ("ProcessInitialization")

            # Smart batches
        batch_size =max (10 ,len (smiles_list )//(self .num_processes *2 ))
        smiles_batches =[smiles_list [i :i +batch_size ]for i in range (0 ,len (smiles_list ),batch_size )]

        # andGenerateImage
        batch_results =list (self .pool .map (parallel_batch_mol_to_images ,smiles_batches ))

        # CollectionImage
        all_images =[]
        total_failed =0 
        for batch_images ,_ ,failed_count in batch_results :
            all_images .extend (batch_images )
            total_failed +=failed_count 

            # BatchCalculateEssays
        representations =[]
        batch_size =128 # Batch


        for i in range (0 ,len (all_images ),batch_size ):
            batch_images =all_images [i :i +batch_size ]

            # Convert totensorBatch, anddiffusionModelTraining is consistent.
            batch_tensors =[]
            for img in batch_images :
                img_tensor =self .transform (img )
                batch_tensors .append (img_tensor )

            if batch_tensors :
                batch_tensor =torch .stack (batch_tensors )

                # Batch, UseanddiffusionSameProcessing
                with torch .no_grad ():
                    device =next (self .encoder .parameters ()).device 
                    batch_tensor =batch_tensor .to (device )

                    # Use it. anddiffusionSameProcessing
                    mean =torch .Tensor ([0.485 ,0.456 ,0.406 ]).to (device ).unsqueeze (0 ).unsqueeze (-1 ).unsqueeze (-1 )
                    std =torch .Tensor ([0.229 ,0.224 ,0.225 ]).to (device ).unsqueeze (0 ).unsqueeze (-1 ).unsqueeze (-1 )

                    # From[-1, 1]Present. [0, 1]（AssumptionstransformPresent. [-1, 1]）
                    if batch_tensor .min ()>=-1.1 and batch_tensor .max ()<=1.1 :# InspectionYes[-1, 1]Scope
                        x_normalized =(batch_tensor +1 )/2 # [-1, 1] -> [0, 1]
                    else :
                        x_normalized =batch_tensor # Assumptions[0, 1]

                        # ImageNetStandardization
                    x_normalized =(x_normalized -mean )/std 
                    # bicubicValuePresent. 224
                    x_normalized =torch .nn .functional .interpolate (x_normalized ,224 ,mode ='bicubic',align_corners =False )

                    batch_reps =self .encoder (x_normalized )# Shapes: [batch_size, 512, 1, 1]

                    # ProcessingBatch: squeezeOne. Dimensions
                    batch_reps =batch_reps .squeeze (-1 ).squeeze (-1 )# Shapes: [batch_size, 512]

                    # EssaysStandardization（anddiffusionconsistent Z-scoreStandardization）
                    batch_reps =normalize_representation (batch_reps )

                    # WillStandardizationEssaysPresent. CPU
                    batch_reps =batch_reps .cpu ()

                    # Make sure. Every one. Essays512Vee.
                    for rep in batch_reps :
                        representations .append (rep )

                        # Failed
        while len (representations )<len (smiles_list ):
            representations .append (torch .zeros (512 ))

        return representations [:len (smiles_list )]


    def clear_cache (self ):
        self .representation_cache .clear ()

    def shutdown_pool (self ):
        if self .pool_manager is not None :
            self .pool_manager .shutdown ()
            self .pool_manager =None 
            self .pool =None 

    def get_cache_size (self ):
        return len (self .representation_cache )


def parse_args ():
    parser =argparse .ArgumentParser (description ='TrainingModel')

    # Data
    parser .add_argument ('--data_path',type =str ,default ='data/MMP/MMP_10k/test_results_deDupFinal_pro.csv',help ='MMPDataCSVDocumentationPath')
    parser .add_argument ('--target_property',type =str ,default ='mw',help ='Target Optimisation Properties')
    parser .add_argument ('--property_label_column',type =str ,default =None ,help ='Optional：DataMediumTagObjectiveIt is. .. Columns，ForMergeDataPressFilter')
    parser .add_argument ('--filter_by_property_label',action ='store_true',help ='If providedColumns，availableYesKeep Only target_property ')
    parser .add_argument ('--property_list',type =str ,default =None ,help ='Columns（Comma Separated）；If not providedFromDataColumnsInference')

    # Model
    parser .add_argument ('--rep_dim',type =int ,default =512 ,help ='EssaysDimensions')
    parser .add_argument ('--hidden_dim',type =int ,default =512 ,help ='LayerDimensions')
    parser .add_argument ('--num_blocks',type =int ,default =3 ,help ='Convert BlocksCount')
    parser .add_argument ('--use_residual',action ='store_true',help ='availableYesUseImpairment')
    parser .add_argument ('--use_attention',action ='store_true',help ='availableYesUseAttention. ')
    parser .add_argument ('--use_ratio_condition',dest ='use_ratio_condition',action ='store_true',
    help ='Use it. ratioConditionInput（Default，Essays）')
    parser .add_argument ('--no_ratio_condition',dest ='use_ratio_condition',action ='store_false',
    help ='Use it. ratioConditionInput')
    parser .set_defaults (use_ratio_condition =None )
    parser .add_argument ('--group_by_property',action ='store_true',help ='PressSample，Every one. batchOrganisationDestination Properties')
    parser .add_argument ('--mask_span',type =int ,default =20 ,help ='Every one. Dimensions（Fixed compartments）')
    parser .add_argument ('--mask_offset',type =int ,default =0 ,help ='StartDimensions（DefaultFrom0Here we go. ）')

    # Training-related
    parser .add_argument ('--batch_size',type =int ,default =32 ,help ='Batch')
    parser .add_argument ('--num_epochs',type =int ,default =100 ,help ='TrainingCount')
    parser .add_argument ('--lr',type =float ,default =1e-4 ,help ='Learning rate')
    parser .add_argument ('--weight_decay',type =float ,default =1e-5 ,help ='Weighted Decomposition')
    parser .add_argument ('--warmup_epochs',type =int ,default =5 ,help ='Count')
    parser .add_argument ('--lr_min_percent',type =float ,default =1.0 ,
    help ='CosineLearning ratePresent. Learning rateIt is. .. ')

    # Weights of loss
    parser .add_argument ('--contrast_weight',type =float ,default =1.0 ,help ='Weights of loss')
    parser .add_argument ('--reconstruction_weight',type =float ,default =0.1 ,help ='Weights of loss')
    parser .add_argument ('--step_weight',type =float ,default =1.0 ,help ='L_step StepWeight')
    parser .add_argument ('--target_property_weight',type =float ,default =1.0 ,help ='L_target_property Weight')
    parser .add_argument ('--invariant_property_weight',type =float ,default =1.0 ,help ='L_invariant_property Weight')
    parser .add_argument ('--temperature',type =float ,default =0.07 ,help ='Count')

    # Loss component switch
    parser .add_argument ('--use_reconstruction_loss',action ='store_true',default =True ,help ='availableYesUseLoss')
    parser .add_argument ('--no_reconstruction_loss',dest ='use_reconstruction_loss',action ='store_false',help ='Use it. Loss')
    parser .add_argument ('--use_step_loss',action ='store_true',default =True ,help ='availableYesUseStepLoss')
    parser .add_argument ('--no_step_loss',dest ='use_step_loss',action ='store_false',help ='Use it. StepLoss')
    parser .add_argument ('--use_target_property_loss',action ='store_true',default =True ,help ='availableYesUseObjectiveLoss')
    parser .add_argument ('--no_target_property_loss',dest ='use_target_property_loss',action ='store_false',help ='Use it. ObjectiveLoss')
    parser .add_argument ('--use_invariant_property_loss',action ='store_true',default =True ,help ='availableYesUseLoss')
    parser .add_argument ('--no_invariant_property_loss',dest ='use_invariant_property_loss',action ='store_false',help ='Use it. Loss')

    # Data
    parser .add_argument ('--negative_sample_size',type =int ,default =5 ,help ='Negative sampleCount')
    parser .add_argument ('--properties_to_preserve',type =str ,default =None ,help ='Columns，Use it. Comma Separated(Like: qed, sa, logp)')
    parser .add_argument ('--correlation_file',type =str ,default =None ,help ='RelevanceOutcomeDocumentationPath')
    parser .add_argument ('--deviation_threshold',type =float ,default =0.1 ,help ='Threshold，For')
    parser .add_argument ('--positive_quality_threshold',type =float ,default =0.2 ,help ='Positive sampleThreshold，ValueIt is. .. As a. .. Positive sample')
    parser .add_argument ('--quality_weight_alpha',type =float ,default =5.0 ,help ='WeightIt is. .. CountCount，0Does that mean you do not have weight?')
    parser .add_argument ('--quality_weight_min',type =float ,default =0.1 ,help ='TrainingWeight，ForAvoidEnd')
    parser .add_argument ('--enable_ratio_dedup',action ='store_true',help ='Use it. ratioWindow（Default）')
    parser .add_argument ('--use_contrastive_loss',action ='store_true',help ='Use it. LossNegative sampleSample（Default）')
    parser .add_argument ('--preferred_sample_weight',type =float ,default =3.0 ,help ='WeightCount')
    parser .add_argument ('--non_preferred_sample_weight',type =float ,default =0.1 ,help ='WeightCount')
    parser .add_argument ('--property_predictor_checkpoint',type =str ,default =None ,help ='MultitaskProjectionsInspectionPath')
    parser .add_argument ('--property_predictor_device',type =str ,default =None ,help ='ProjectionsRunEquipment（DefaultTrainingEquipment）')

    # Other
    parser .add_argument ('--device',type =str ,default ='cuda',help ='Equipment')
    parser .add_argument ('--num_workers',type =int ,default =mp .cpu_count (),help ='DataLoadCount（DefaultUseCPUCore）')
    parser .add_argument ('--save_dir',type =str ,default ='train_optimization_MMP/checkpoints',help ='ModelSavePath')
    parser .add_argument ('--save_interval',type =int ,default =10 ,help ='ModelSave')
    parser .add_argument ('--resume',type =str ,default =None ,help ='FromInspectionRestoring trainingIt is. .. Path')
    parser .add_argument (
    '--resume_model_only',
    action ='store_true',
    help ='LoadModelWeight，Ignore the optimiser/Scheduler Status，From1One. epochPressused inLearning rateHere we go. '
    )
    parser .add_argument ('--seed',type =int ,default =42 ,help ='Random Feeds')

    # TrainingEncoder
    parser .add_argument ('--encoder_path',type =str ,default ='checkpoints/pretrained_enc_ckpts/CGIP/CGIP.pth',help ='CGIPEncoderWeightPath')

    # EssaysCalculate
    parser .add_argument ('--repr_num_processes',type =int ,default =20 ,help ='EssaysCalculateIt is. .. ProcessCount')
    parser .add_argument ('--repr_batch_size',type =int ,default =32 ,help ='EssaysCalculateBatch')

    # LoadEssaysCache
    parser .add_argument ('--precomputed_reps_path',type =str ,default =None ,help ='CalculateIt is. .. SMILES->EssaysCachePath(.pt)')

    return parser .parse_args ()


def set_seed (seed ):
    torch .manual_seed (seed )
    torch .cuda .manual_seed_all (seed )
    np .random .seed (seed )
    import random 
    random .seed (seed )
    torch .backends .cudnn .deterministic =True 


def get_lr (optimizer ):
    for param_group in optimizer .param_groups :
        return param_group ['lr']


def save_checkpoint (model ,optimizer ,epoch ,loss ,save_path ):
    checkpoint ={
    'epoch':epoch ,
    'model_state_dict':model .state_dict (),
    'optimizer_state_dict':optimizer .state_dict (),
    'loss':loss ,
    }
    torch .save (checkpoint ,save_path )
    print (f"ModelSavePresent. : {save_path}")


def load_checkpoint (model ,optimizer ,checkpoint_path ,load_optimizer :bool =True ,map_location =None ):
    checkpoint =torch .load (checkpoint_path ,map_location =map_location )
    model .load_state_dict (checkpoint ['model_state_dict'])
    if load_optimizer and optimizer is not None and 'optimizer_state_dict'in checkpoint :
        optimizer .load_state_dict (checkpoint ['optimizer_state_dict'])
    return checkpoint .get ('epoch',0 ),checkpoint .get ('loss')


def compute_dynamic_representations (batch ,dynamic_repr_calculator ,device ):
    """availableBatchCalculateEssays"""

    source_smiles =list (batch ['source_smiles'])
    has_negatives ='negative_smiles'in batch and len (batch ['negative_smiles'])>0 
    negative_smiles_list =[]
    if has_negatives :
        for neg_group in batch ['negative_smiles']:
            negative_smiles_list .append (list (neg_group ))

    all_smiles =set (source_smiles )
    if has_negatives :
        for neg_list in negative_smiles_list :
            all_smiles .update (neg_list )

    all_smiles =list (all_smiles )

    start_time =time .time ()
    all_representations =dynamic_repr_calculator .get_batch_representations (all_smiles )
    end_time =time .time ()
    print ('dynamic_repr_calculator. get_batch_representations time is: ',end_time -start_time )

    smiles_to_repr ={smiles :repr for smiles ,repr in zip (all_smiles ,all_representations )}

    source_reps =torch .stack ([smiles_to_repr [smiles ]for smiles in source_smiles ])
    batch ['source_rep']=source_reps .to (device )

    if has_negatives and any (len (lst )>0 for lst in negative_smiles_list ):
        negative_reps =[]
        for neg_list in negative_smiles_list :
            if len (neg_list )==0 :
                continue 
            neg_reps =torch .stack ([smiles_to_repr [smiles ]for smiles in neg_list ])
            negative_reps .append (neg_reps )
        if negative_reps :
            batch ['negatives']=torch .stack (negative_reps ).to (device )

    return batch 


def compute_cached_representations (batch ,rep_cache :dict ,device ):
    """FromLoadIt is. .. .pt MediumavailableBatchLoadEssays"""
    source_smiles =batch .get ('source_smiles',[])
    negative_smiles_list =batch .get ('negative_smiles',[])

    all_smiles =set (source_smiles )
    if negative_smiles_list :
        for neg_list in negative_smiles_list :
            all_smiles .update (neg_list )

    smiles_to_repr ={}
    zero =torch .zeros (512 ,dtype =torch .float32 )
    for smi in all_smiles :
        rep =rep_cache .get (smi )
        if isinstance (rep ,torch .Tensor ):
            smiles_to_repr [smi ]=rep .to (dtype =torch .float32 ,device ='cpu')
        else :
            smiles_to_repr [smi ]=zero 

    source_reps =torch .stack ([smiles_to_repr [smi ]for smi in source_smiles ])
    batch ['source_rep']=source_reps .to (device )

    if negative_smiles_list and any (len (lst )>0 for lst in negative_smiles_list ):
        negative_reps =[]
        for neg_list in negative_smiles_list :
            neg_reps =torch .stack ([smiles_to_repr [smi ]for smi in neg_list ])
            negative_reps .append (neg_reps )
        if negative_reps :
            negatives =torch .stack (negative_reps )
            batch ['negatives']=negatives .to (device )
    return batch 


def train_epoch (
model ,
dataloader ,
optimizer ,
criterion ,
device ,
epoch ,
dynamic_repr_calculator =None ,
rep_cache =None ,
time_log_file =None ,
property_predictor =None ,
property_task_configs =None ,
property_normalizers =None ,
target_property_name =None ,
properties_to_preserve =None ,
property_names_map =None ,
):
    """TrainingOne. epoch，EssaysCalculateandRecords"""
    model .train ()
    total_loss =0 
    loss_components ={
    'total_weighted':0.0 ,
    'total_raw':0.0 ,
    'target_property_raw':0.0 ,
    'target_property_weighted':0.0 ,
    'invariant_raw':0.0 ,
    'invariant_weighted':0.0 
    }


    pbar =tqdm (dataloader ,desc =f'Epoch {epoch}')

    for batch_idx ,batch in enumerate (pbar ):

    # According toAvailabilityFillEssays
        need_fill ='source_rep'not in batch 
        if need_fill :
            if rep_cache is not None :
                start_time =time .time ()
                batch =compute_cached_representations (batch ,rep_cache ,device )
                end_time =time .time ()
                # print('compute_cached_representations time is: ', end_time-start_time)
            elif dynamic_repr_calculator is not None :
                start_time =time .time ()
                batch =compute_dynamic_representations (batch ,dynamic_repr_calculator ,device )
                end_time =time .time ()
                print ('compute_dynamic_representations time is: ',end_time -start_time )

        loss ,loss_dict =train_step (
        model ,
        batch ,
        criterion ,
        device ,
        property_predictor =property_predictor ,
        property_task_configs =property_task_configs ,
        property_normalizers =property_normalizers ,
        target_property_name =target_property_name ,
        preserve_properties =properties_to_preserve if properties_to_preserve else None ,
        property_names_map =property_names_map ,
        )

        optimizer .zero_grad ()
        loss .backward ()

        # Gradient Crop
        torch .nn .utils .clip_grad_norm_ (model .parameters (),max_norm =1.0 )
        optimizer .step ()

        # Cumulative losses
        total_loss +=loss .item ()
        for key in loss_components :
            loss_components [key ]+=loss_dict .get (key ,0.0 )

            # ProgressArticle
        postfix_dict ={
        'loss':f'{loss. item(): .4f}',
        'lr':f'{get_lr(optimizer): .6f}'
        }

        pbar .set_postfix (postfix_dict )

        # Records（Like）

        # CalculateAverage loss
    avg_loss =total_loss /len (dataloader )
    for key in loss_components :
        loss_components [key ]/=len (dataloader )

    return avg_loss ,loss_components 




def main ():
    args =parse_args ()

    property_list_clean =[]
    if args .property_list :
        property_list_clean =[p .strip ()for p in args .property_list .split (', ')if p .strip ()]
        args .property_list =', '.join (property_list_clean )

        # If providedcolumns property_list MediumOrganisationandDefault target_property consistent, Autoreplace, AvoidDefault mw columnsError
    if args .property_label_column and property_list_clean :
        if args .target_property not in property_list_clean :
            print (
            f"DetectedColumns {args. property_label_column}，Will target_property From {args. target_property} available {property_list_clean[0]}"
            )
            args .target_property =property_list_clean [0 ]

    if args .use_ratio_condition is None :
        args .use_ratio_condition =False 

        # Reservation only L_target_property and L_invariant_property, The rest of the losses are shut down.
    args .use_contrastive_loss =False 
    args .use_step_loss =False 
    args .use_reconstruction_loss =False 
    args .negative_sample_size =0 
    args .contrast_weight =0.0 
    args .step_weight =0.0 
    args .reconstruction_weight =0.0 
    print (f"Ratio conditioning enabled: {args. use_ratio_condition}")

    # Random Feeds
    set_seed (args .seed )

    # CreateSave
    os .makedirs (args .save_dir ,exist_ok =True )

    # Equipment
    device =torch .device (args .device if torch .cuda .is_available ()else 'cpu')
    print (f"Use of equipment: {device}")

    pretrained_encoder =models_pretrained_enc .__dict__ ['CGIP_image_model']()
    pretrained_encoder =models_pretrained_enc .load_pretrained_CGIP_image_ckpt (
    pretrained_encoder ,args .encoder_path 
    )
    pretrained_encoder .to (device )
    pretrained_encoder .eval ()
    for param in pretrained_encoder .parameters ():
        param .requires_grad =False 

        # CalculateCacheProcessLoad, TrainingPress
    dynamic_repr_calculator =None 
    rep_cache =None 
    if args .precomputed_reps_path is not None and os .path .exists (args .precomputed_reps_path ):
        path_lower =args .precomputed_reps_path .lower ()
        if path_lower .endswith ('.pt'):
            print (f"LoadCalculateEssaysCache(.pt): {args. precomputed_reps_path}")
            rep_cache =torch .load (args .precomputed_reps_path ,map_location ='cpu')
            print (f" ✓ Loaded {len(rep_cache): ,} ArticlePresent. （Process）")
        else :
            raise ValueError (f"Support only.pt CalculateCache，Present. : {args. precomputed_reps_path}")
    else :
    # Cache, Use it. Calculate
        dynamic_repr_calculator =DynamicRepresentationCalculator (
        encoder_model =pretrained_encoder ,
        num_processes =args .repr_num_processes ,
        enable_multiprocess =args .repr_num_processes >1 
        )


        # Processingproperties_to_preserveCount
    properties_to_preserve =None 
    if args .properties_to_preserve :
        properties_to_preserve =[prop .strip ()for prop in args .properties_to_preserve .split (', ')if prop .strip ()]
    elif property_list_clean :
        properties_to_preserve =property_list_clean 
    properties_to_preserve_list =list (dict .fromkeys (properties_to_preserve ))if properties_to_preserve else []

    # CreateDataLoad（Dataset Again. Cache, BackSMILES）
    train_dataloader =create_dataloader (
    csv_path =args .data_path ,
    encoder_model =None ,
    batch_size =args .batch_size ,
    shuffle =False ,
    num_workers =args .num_workers ,
    precomputed_reps_path =None ,
    target_property =args .target_property ,
    property_list =property_list_clean ,
    negative_sample_size =args .negative_sample_size ,
    properties_to_preserve =properties_to_preserve ,
    correlation_file =args .correlation_file ,
    deviation_threshold =args .deviation_threshold ,
    positive_quality_threshold =args .positive_quality_threshold ,
    quality_weight_alpha =args .quality_weight_alpha ,
    quality_weight_min =args .quality_weight_min ,
    enable_ratio_dedup =args .enable_ratio_dedup ,
    use_contrastive_loss =args .use_contrastive_loss ,
    preferred_sample_weight =args .preferred_sample_weight ,
    non_preferred_sample_weight =args .non_preferred_sample_weight ,
    property_label_column =args .property_label_column ,
    filter_by_property_label =args .filter_by_property_label ,
    group_by_property =args .group_by_property ,
    )
    property_names_map =getattr (train_dataloader .dataset ,"property_vocab",property_list_clean or [args .target_property ])

    # FilterValidity, Every one. ExpertsTrainingData
    dataset =train_dataloader .dataset 
    if getattr (dataset ,"property_labels",None )is not None :
        labels_np =dataset .property_labels .to_numpy ()
        valid_idx =getattr (dataset ,"valid_indices",np .arange (len (labels_np )))
        labels_valid =labels_np [valid_idx ]
        print ("ValidityCount（Filter）: ")
        total_count =len (labels_valid )
        for prop in property_names_map :
            cnt =int ((labels_valid ==prop ).sum ())
            print (f" - {prop}: {cnt}")
        print (f" Total: {total_count}")
    else :
        valid_idx =getattr (dataset ,"valid_indices",np .arange (len (dataset )))
        print (f"ValidityCount（Columns）: {len(valid_idx)}")

    property_predictor =None 
    property_task_configs =None 
    property_normalizers =None 
    if args .property_predictor_checkpoint :
        predictor_device =torch .device (args .property_predictor_device )if args .property_predictor_device else device 
        print (f"LoadProjections: {args. property_predictor_checkpoint}")
        payload ,property_predictor ,property_normalizers =load_property_predictor_checkpoint (
        args .property_predictor_checkpoint ,
        predictor_device 
        )
        property_task_configs =payload .get ('task_configs',{})
        predictor_properties =payload .get ('properties',[])
        required_properties =sorted (set (property_names_map +properties_to_preserve_list ))
        missing =[prop for prop in required_properties if prop not in predictor_properties ]
        if missing :
            raise ValueError (
            f"ProjectionsMissingBelow: {', '. join(missing)}，TrainingorIt is. .. Inspection"
            )
        property_predictor .to (device )
        property_predictor .eval ()
        for param in property_predictor .parameters ():
            param .requires_grad_ (False )
    else :
        raise ValueError ("used inReservation only L_target_property and L_invariant_property，It has to be supplied. property_predictor_checkpoint。")

        # CreateModel
    print ("CreateModel. .. ")
    model =ConditionalRepAdjuster (
    rep_dim =args .rep_dim ,
    hidden_dim =args .hidden_dim ,
    num_blocks =args .num_blocks ,
    use_residual =args .use_residual ,
    use_attention =args .use_attention ,
    use_condition =args .use_ratio_condition ,
    property_names =property_names_map ,
    mask_span =args .mask_span ,
    mask_offset =args .mask_offset ,
    )

    model .to (device )
    print (f"ModelCount: {sum(p. numel() for p in model. parameters() if p. requires_grad): ,}")

    # CreateOptimizer
    optimizer =optim .AdamW (
    model .parameters (),
    lr =args .lr ,
    weight_decay =args .weight_decay 
    )

    # CreateLearning rate
    min_lr_ratio =min (max (args .lr_min_percent ,0.0 ),100.0 )/100.0 
    scheduler =optim .lr_scheduler .CosineAnnealingLR (
    optimizer ,
    T_max =args .num_epochs ,
    eta_min =args .lr *min_lr_ratio 
    )

    # Restoring training（If Inspection）
    start_epoch =1 
    if args .resume :
        if os .path .exists (args .resume ):
            print (f"🚀 FromInspectionRestoring training: {args. resume}")
            resume_epoch ,resume_loss =load_checkpoint (
            model ,
            optimizer ,
            args .resume ,
            load_optimizer =not args .resume_model_only ,
            map_location =device ,
            )
            if args .resume_model_only :
                start_epoch =1 
                print ("LoadModelWeight，Optimizer/Pressused inCountHere we go. Training。")
            else :
                start_epoch =resume_epoch +1 
                loss_display =resume_loss if resume_loss is not None else float ('nan')
                print (f"Restoring trainingFrom {start_epoch} One. epochHere we go. ，Previous loss: {loss_display: .4f}")

                # Learning rate
                for _ in range (resume_epoch ):
                    scheduler .step ()
        else :
            print (f"⚠️ CKPTDocumentationavailable: {args. resume}")
            print ("FromHere we go. Training. .. ")

            # CreateLossCount
    criterion =OptimizationLoss (
    target_property_weight =args .target_property_weight ,
    invariant_property_weight =args .invariant_property_weight ,
    use_target_property_loss =args .use_target_property_loss ,
    use_invariant_property_loss =args .use_invariant_property_loss ,
    )

    # Save
    args .property_names =property_names_map 
    args .mask_span =args .mask_span 
    config_path =os .path .join (args .save_dir ,'config.json')
    with open (config_path ,'w',encoding ='utf-8')as f :
        json .dump (vars (args ),f ,indent =4 ,ensure_ascii =False )
    print (f"SavePresent. : {config_path}")

    # CreateDocumentationPath
    time_log_path =os .path .join (args .save_dir ,'training_time_log.csv')
    print (f"WillSavePresent. : {time_log_path}")

    # CreatelossRecordsDocumentationPath
    loss_log_path =os .path .join (args .save_dir ,'training_loss_log.txt')
    print (f"LossWillSavePresent. : {loss_log_path}")

    # CreatelossRecordsCSVDocumentationPath
    metrics_csv_path =os .path .join (args .save_dir ,'training_metrics.csv')
    print (f"Loss CSVWillSavePresent. : {metrics_csv_path}")

    # InitializationlossRecordsDocumentation, Write to table header
    with open (loss_log_path ,'w',encoding ='utf-8')as f :
        f .write ("TrainingLossRecords\n")
        f .write ("="*120 +"\n")
        f .write (f"TrainingHere we go. : {datetime. now(). strftime('%Y-%m-%d %H: %M: %S')}\n")
        f .write (f"ModelCount: {sum(p. numel() for p in model. parameters() if p. requires_grad): ,}\n")
        f .write (f"Batch: {args. batch_size}\n")
        f .write (f"Learning rate: {args. lr}\n")
        f .write (f"Learning rate: {args. lr_min_percent}\n")
        f .write ("-"*120 +"\n")
        f .write (
        "Format: Epoch | AvgTotal(weighted) | AvgTotal(raw) | Target(raw) | Target(weighted) | "
        "Invariant(raw) | Invariant(weighted) | Learning rate\n"
        )
        f .write ("-"*120 +"\n")
    with open (metrics_csv_path ,'w',newline ='',encoding ='utf-8')as csv_file :
        csv_writer =csv .writer (csv_file )
        csv_writer .writerow ([
        "epoch",
        "avg_total_weighted",
        "avg_total_raw",
        "target_property_raw",
        "target_property_weighted",
        "invariant_raw",
        "invariant_weighted",
        "learning_rate"
        ])

        # Training
    print ("Here we go. Training. .. ")
    best_loss =float ('inf')

    for epoch in range (start_epoch ,args .num_epochs +1 ):
        avg_loss ,loss_components =train_epoch (
        model ,train_dataloader ,optimizer ,criterion ,device ,epoch ,
        dynamic_repr_calculator =dynamic_repr_calculator ,
        rep_cache =rep_cache ,
        time_log_file =time_log_path ,
        property_predictor =property_predictor ,
        property_task_configs =property_task_configs ,
        property_normalizers =property_normalizers ,
        target_property_name =args .target_property ,
        properties_to_preserve =properties_to_preserve_list ,
        property_names_map =property_names_map ,
        )

        # Learning rate
        scheduler .step ()

        # Tag（Like）

        print (f"\nEpoch {epoch}/{args. num_epochs}")
        print (f"Average loss: {avg_loss: .4f}")
        print (f"Various losses: {loss_components}")

        current_lr =get_lr (optimizer )

        # RecordslossPresent. txtDocumentation
        with open (loss_log_path ,'a',encoding ='utf-8')as f :
            f .write (
            f"{epoch: 4d} | {avg_loss: 8.4f} | {loss_components['total_raw']: 8.4f} | "
            f"{loss_components['target_property_raw']: 8.4f} | {loss_components['target_property_weighted']: 8.4f} | "
            f"{loss_components['invariant_raw']: 8.4f} | {loss_components['invariant_weighted']: 8.4f} | "
            f"{current_lr: .2e}\n"
            )
        with open (metrics_csv_path ,'a',newline ='',encoding ='utf-8')as csv_file :
            csv_writer =csv .writer (csv_file )
            csv_writer .writerow ([
            epoch ,
            round (avg_loss ,6 ),
            round (loss_components ['total_raw'],6 ),
            round (loss_components ['target_property_raw'],6 ),
            round (loss_components ['target_property_weighted'],6 ),
            round (loss_components ['invariant_raw'],6 ),
            round (loss_components ['invariant_weighted'],6 ),
            current_lr 
            ])


            # SaveModel
        if epoch %args .save_interval ==0 :
            checkpoint_path =os .path .join (
            args .save_dir ,
            f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint (model ,optimizer ,epoch ,avg_loss ,checkpoint_path )

            # SaveModel
        if avg_loss <best_loss :
            best_loss =avg_loss 
            best_path =os .path .join (args .save_dir ,'best_model.pth')
            save_checkpoint (model ,optimizer ,epoch ,avg_loss ,best_path )
            print (f"SaveModel，Loss: {best_loss: .4f}")

            # SaveModel
    final_path =os .path .join (args .save_dir ,'final_model.pth')
    save_checkpoint (model ,optimizer ,args .num_epochs ,avg_loss ,final_path )

    # lossRecordsDocumentationMediumAddTraining
    with open (loss_log_path ,'a',encoding ='utf-8')as f :
        f .write ("-"*60 +"\n")
        f .write (f"Training: {datetime. now(). strftime('%Y-%m-%d %H: %M: %S')}\n")
        f .write (f"TrainingCount: {args. num_epochs}\n")
        f .write (f"Loss: {best_loss: .4f}\n")
        f .write (f"Loss: {avg_loss: .4f}\n")
        f .write ("="*60 +"\n")


    print ("Training！")


if __name__ =='__main__':
    main ()
