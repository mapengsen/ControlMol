import os 
import sys 
import json 
import random 
import torch 
import argparse 
import numpy as np 
import torch .nn .functional as F 
from rdkit import Chem 
from rdkit .Chem import Draw 
from PIL import Image 
import pandas as pd 
import cv2 
import time 
from omegaconf import OmegaConf 
from typing import Dict ,Optional 

# AddPath
sys .path .append (os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))

from train_optimization_MMP .model import ConditionalRepAdjuster 
from train_optimization_MMP .train_property_predictor import (
load_checkpoint as load_property_predictor_checkpoint ,
)
import pretrained_enc .models_pretrained_enc as models_pretrained_enc 
import torchvision .transforms as transforms 
from pixel_generator .ldm .util import instantiate_from_config 
from pixel_generator .ldm .models .diffusion .ddim import DDIMSampler 


def scale_to_range (x ):
    """Will[0, 1]Zoom to[-1, 1]"""
    return x *2.0 -1.0 


def normalize_representation (rep ):
    """anddiffusionModelTrainingconsistentZ-scoreStandardization"""
    rep_std =torch .std (rep ,dim =1 ,keepdim =True )
    rep_mean =torch .mean (rep ,dim =1 ,keepdim =True )
    rep_std =torch .clamp (rep_std ,min =1e-8 )# Avoiding the elimination of zeroes
    return (rep -rep_mean )/rep_std 


def safe_torch_load (path ,map_location ="cpu"):
    """Torch 2.6 Default weights_only=True，Promise. CompatibilityOldcheckpoint。"""
    kwargs =dict (map_location =map_location )
    try :
        return torch .load (path ,weights_only =False ,**kwargs )
    except TypeError :
    # OldVersion torch The, Present.
        return torch .load (path ,**kwargs )


def extract_into_tensor (a ,t ,x_shape ):
    """Fromtensor aMediumAccording toSteptExtract value，andreshapePresent. x_shapeIt is. .. batchDimensions"""
    b ,*_ =t .shape 
    # Make sure. IndexScope
    t_clamped =torch .clamp (t ,0 ,a .shape [0 ]-1 )
    # Make sure. aandtEquipment
    if a .device !=t .device :
        a =a .to (t .device )
    out =a .gather (-1 ,t_clamped )
    return out .reshape (b ,*((1 ,)*(len (x_shape )-1 )))


def q_sample_with_noise (x_start ,t ,sqrt_alphas_cumprod ,sqrt_one_minus_alphas_cumprod ,noise =None ,noise_strength =1.0 ):
    """ According toDDPMIt is. .. forwardAddNoise Args: x_start: Original Image/Essays t: Step sqrt_alphas_cumprod: sqrt(alpha_cumprod) sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_cumprod) noise: Noise，If. .. availableNoneGenerateNoise noise_strength: Noise """
    if noise is None :
        noise =torch .randn_like (x_start )

        # Noise
    noise =noise *noise_strength 

    return (extract_into_tensor (sqrt_alphas_cumprod ,t ,x_start .shape )*x_start +
    extract_into_tensor (sqrt_one_minus_alphas_cumprod ,t ,x_start .shape )*noise )


def set_global_seed (seed :int ):
    """Set global random torrents，Make sure. Sample。"""
    random .seed (seed )
    np .random .seed (seed )
    torch .manual_seed (seed )
    torch .cuda .manual_seed_all (seed )
    torch .backends .cudnn .benchmark =False 
    torch .backends .cudnn .deterministic =True 


class MolecularOptimizerWithGeneration :
    """Molecular Optimizer，OrganisationEssaysandImageGenerate"""

    def __init__ (
    self ,
    optimizer_model_path :str ,
    diffusion_model_config :str ,
    diffusion_model_path :str ,
    encoder_path :str ,
    device :str ='cuda',
    use_attention :bool =True ,
    use_residual :bool =True ,
    rep_dim :int =512 ,
    hidden_dim :int =512 ,
    num_blocks :int =3 ,
    use_ratio_condition :bool =False ,
    property_names :Optional [list ]=None ,
    target_property :Optional [str ]=None ,
    property_masks_span :int =20 ,
    property_mask_offset :int =0 ,
    property_predictor_checkpoint :Optional [str ]=None ,
    property_predictor_device :Optional [str ]=None ,
    ):
        self .device =torch .device (device if torch .cuda .is_available ()else 'cpu')
        self .use_ratio_condition =use_ratio_condition 
        self .property_names =property_names or [target_property or "default"]
        self .property_to_idx ={name :idx for idx ,name in enumerate (self .property_names )}
        self .target_property =target_property or self .property_names [0 ]
        self .property_masks_span =int (property_masks_span )
        self .property_mask_offset =int (property_mask_offset )

        # 1. LoadCGIPEncoder
        print ("Loading CGIP encoder. .. ")
        self .encoder =models_pretrained_enc .__dict__ ['CGIP_image_model']()
        self .encoder =models_pretrained_enc .load_pretrained_CGIP_image_ckpt (
        self .encoder ,encoder_path 
        )
        self .encoder .to (self .device )
        self .encoder .eval ()

        # 2. Image Change - anddiffusionConsistency in training
        self .transform =transforms .Compose ([
        transforms .Resize ((256 ,256 )),
        transforms .ToTensor (),
        transforms .Lambda (scale_to_range )# UseCountlambda
        ])

        # 3. LoadModel
        print ("Loading optimization model. .. ")
        optimizer_checkpoint =safe_torch_load (optimizer_model_path ,map_location =self .device )
        self .optimizer_model =ConditionalRepAdjuster (
        rep_dim =rep_dim ,
        hidden_dim =hidden_dim ,
        num_blocks =num_blocks ,
        use_attention =use_attention ,
        use_residual =use_residual ,
        use_condition =self .use_ratio_condition ,
        property_names =self .property_names ,
        mask_span =self .property_masks_span ,
        mask_offset =self .property_mask_offset ,
        )
        self .optimizer_model .load_state_dict (optimizer_checkpoint ['model_state_dict'])
        self .optimizer_model .to (self .device )
        self .optimizer_model .eval ()

        # 4. LoadModel
        print ("Loading diffusion model. .. ")
        config =OmegaConf .load (diffusion_model_config )
        self .diffusion_model =instantiate_from_config (config .model )

        # LoadModelWeight
        diffusion_checkpoint =safe_torch_load (diffusion_model_path ,map_location =self .device )
        if 'model'in diffusion_checkpoint :
            self .diffusion_model .load_state_dict (diffusion_checkpoint ['model'],strict =False )
        elif 'state_dict'in diffusion_checkpoint :
            self .diffusion_model .load_state_dict (diffusion_checkpoint ['state_dict'],strict =False )
        else :
            self .diffusion_model .load_state_dict (diffusion_checkpoint ,strict =False )

        self .diffusion_model .to (self .device )
        self .diffusion_model .eval ()

        # 5. Loadprediction（Optional）
        self .property_predictor =None 
        self .property_normalizers ={}
        self .property_task_configs ={}
        self .property_property_names =[]
        if property_predictor_checkpoint :
            predictor_device =torch .device (property_predictor_device )if property_predictor_device else self .device 
            print ("Loading property predictor. .. ")
            payload ,predictor_model ,normalizers =load_property_predictor_checkpoint (
            property_predictor_checkpoint ,
            predictor_device 
            )
            predictor_model .to (self .device )
            predictor_model .eval ()
            for param in predictor_model .parameters ():
                param .requires_grad_ (False )
            self .property_predictor =predictor_model 
            self .property_normalizers =normalizers 
            self .property_task_configs =payload .get ('task_configs',{})
            self .property_property_names =payload .get ('properties',[])
            print (
            "Property predictor heads: "
            +(", ".join (self .property_property_names )if self .property_property_names else "unknown")
            )
        else :
            print ("Property predictor not provided; property estimates will be skipped. ")

            # 6. InitializationDDIMSampler
        self .ddim_sampler =DDIMSampler (self .diffusion_model )

        # InitializationDDIMMake sure. CountUse it.
        print ("Initializing DDIM scheduler. .. ")
        self .ddim_sampler .make_schedule (ddim_num_steps =50 ,ddim_eta =0.0 ,verbose =False )

        print ("All models loaded successfully!")

    def mol_to_image (self ,smiles :str )->Image .Image :
        """WillSMILESConvert toImage"""
        mol =Chem .MolFromSmiles (smiles )
        if mol is None :
            raise ValueError (f"Invalid SMILES: {smiles}")

        img =Draw .MolToImage (mol ,size =(256 ,256 ))# anddiffusionIt is the same training.
        return img 

    def get_molecular_representation (self ,smiles :str )->torch .Tensor :
        """GetIt is. .. Essays"""
        # GenerateImage
        img =self .mol_to_image (smiles )

        # Convert totensor, andConsistency in training
        img_tensor =self .transform (img ).unsqueeze (0 ).to (self .device )

        # GetEssays, UseanddiffusionSameProcessing
        with torch .no_grad ():
        # Use it. anddiffusionSameProcessing
            mean =torch .Tensor ([0.485 ,0.456 ,0.406 ]).to (self .device ).unsqueeze (0 ).unsqueeze (-1 ).unsqueeze (-1 )
            std =torch .Tensor ([0.229 ,0.224 ,0.225 ]).to (self .device ).unsqueeze (0 ).unsqueeze (-1 ).unsqueeze (-1 )

            # From[-1, 1]Present. [0, 1]（anddiffusionTraining is consistent. ）
            x_normalized =(img_tensor +1 )/2 # [-1, 1] -> [0, 1]
            # ImageNetStandardization
            x_normalized =(x_normalized -mean )/std 
            # bicubicValuePresent. 224（anddiffusionTraining is consistent. ）
            x_normalized =torch .nn .functional .interpolate (x_normalized ,224 ,mode ='bicubic',align_corners =False )

            rep =self .encoder (x_normalized )# (1, 512, 1, 1)

            # squeezeOne. Dimensions
            rep =rep .squeeze (-1 ).squeeze (-1 )# (1, 512)

            # Standardization（anddiffusionconsistent Z-scoreStandardization）
            rep =normalize_representation (rep )

            # Get rid of it. batchDimensions
            rep =rep .squeeze (0 )# (512,)

        return rep 

    def optimize_representation (
    self ,
    smiles :str ,
    target_ratio :Optional [float ]=None ,
    target_property :Optional [str ]=None ,
    )->torch .Tensor :
        """ Essays Args: smiles: InputIt is. .. SMILES target_ratio: Objective；used inModelUse it. Condition target_property: Objective，ForSelectionExperts Returns: Essays """
        # GetEssays
        source_rep =self .get_molecular_representation (smiles )

        # Essays
        with torch .no_grad ():
            ratio_tensor =None 
            if self .use_ratio_condition :
                if target_ratio is None :
                    raise ValueError ("target_ratio must be provided when ratio conditioning is enabled. ")
                ratio_tensor =torch .tensor ([target_ratio ],dtype =torch .float32 ).to (self .device )
            prop_name =target_property or self .target_property 
            prop_idx =self .property_to_idx .get (prop_name ,0 )
            property_ids =torch .tensor ([prop_idx ],dtype =torch .long ,device =self .device )
            adjusted_rep =self .optimizer_model (source_rep .unsqueeze (0 ),ratio_tensor ,property_ids =property_ids )
            adjusted_rep =adjusted_rep .squeeze (0 )

        return adjusted_rep 

    def _predict_properties (self ,representation :torch .Tensor )->Optional [Dict [str ,Dict [str ,float ]]]:
        """UseProjectionsInferenceused inEssaysIt is. .. 。"""
        if self .property_predictor is None :
            return None 

        with torch .no_grad ():
            outputs =self .property_predictor (representation .unsqueeze (0 ))

        predictions :Dict [str ,Dict [str ,float ]]={}
        for prop ,logits in outputs .items ():
            cfg =self .property_task_configs .get (prop ,{})
            prop_type =cfg .get ('type','regression')
            entry :Dict [str ,float ]={}

            if prop_type =='classification':
                probability =torch .sigmoid (logits .squeeze ()).item ()
                entry ['prob']=probability 
                entry ['label']=float (probability >=0.5 )
            else :
                value_tensor =logits .squeeze ()
                normalizer =self .property_normalizers .get (prop )
                if normalizer is not None :
                    value_tensor =normalizer .denormalize (value_tensor )
                entry ['value']=value_tensor .item ()

            predictions [prop ]=entry 

        return predictions 

    def generate_images_from_representation (
    self ,
    representation :torch .Tensor ,
    original_smiles :str ,
    num_images :int =5 ,
    ddim_steps :int =50 ,
    eta :float =1.0 ,
    diffusion_batch_size :int =1 ,
    noise_timestep :int =20 ,
    use_pure_noise :bool =False ,
    noise_strength :float =1.0 
    )->list :
        """ FromEssaysGenerateImage Args: representation: Essays original_smiles: Original MolecularSMILES（ForGetx0） num_images: GenerateImageCount ddim_steps: DDIMSampleStepCount eta: DDIMSampleCount diffusion_batch_size: ModelBatch Size noise_timestep: NoiseIt is. .. Step（0-1000） use_pure_noise: availableYesUseNoiseAs a. .. Start noise_strength: Noise Returns: GenerateIt is. .. ImageColumns """
        print (f"Generating with noise_timestep={noise_timestep}, noise_strength={noise_strength}, use_pure_noise={use_pure_noise}")

        # GetOriginal MolecularImageAs a x0
        original_img =self .mol_to_image (original_smiles )
        original_img_tensor =self .transform (original_img ).unsqueeze (0 ).to (self .device )

        with torch .no_grad ():
        # EncodingOriginal ImagePresent.
            encoder_posterior =self .diffusion_model .encode_first_stage (original_img_tensor )
            x0 =self .diffusion_model .get_first_stage_encoding (encoder_posterior ).detach ()

            # Condition（EssaysAgain. Z-Score, andTraining is consistent. ）
            if representation .dim ()==1 :
                rep_batch =representation .unsqueeze (0 )# [512] -> [1, 512]
            else :
                rep_batch =representation # [B, 512]

            rep_batch =normalize_representation (rep_batch )
            cond =rep_batch .unsqueeze (1 )# [B, 512] -> [B, 1, 512]

            # GenerateMultipleImage, Processing
            generated_images =[]
            shape =[4 ,32 ,32 ]# Shapes

            # CalculateOne. batch
            num_batches =(num_images +diffusion_batch_size -1 )//diffusion_batch_size 

            for batch_idx in range (num_batches ):
            # Calculatebatch
                start_idx =batch_idx *diffusion_batch_size 
                end_idx =min (start_idx +diffusion_batch_size ,num_images )
                current_batch_size =end_idx -start_idx 

                # batchConditionandx0
                if current_batch_size ==1 :
                    batch_cond =cond 
                    batch_x0 =x0 
                else :
                    batch_cond =cond .repeat (current_batch_size ,1 ,1 )
                    batch_x0 =x0 .repeat (current_batch_size ,1 ,1 ,1 )

                    # DDIMSample - OrganisationEncodingAs a, and start_timestep YesNoise
                if use_pure_noise :
                    print ('NoiseAs a. .. .. .. .. .. .. .. .. .')
                    # NoiseAs a
                    x_random =torch .randn ((current_batch_size ,*shape ),device =self .device )*noise_strength 
                    sampled_latents ,_ =self .ddim_sampler .sample (
                    ddim_steps ,
                    conditioning =batch_cond ,
                    batch_size =current_batch_size ,
                    shape =shape ,
                    eta =eta ,
                    verbose =False ,
                    x_T =x_random ,
                    )
                else :
                    print ('Use the encoded subvaria as a starting point；If noise_timestep>0，availableSampler x0 Noise. .. .. .. .. .. .. .. .. ')
                    # Use the encoded subvaria as a starting point; If noise_timestep>0, Sampler x0 Noise
                    sampled_latents ,_ =self .ddim_sampler .sample (
                    ddim_steps ,
                    conditioning =batch_cond ,
                    batch_size =current_batch_size ,
                    shape =shape ,
                    eta =eta ,
                    verbose =False ,
                    x0 =batch_x0 ,
                    start_timestep =int (noise_timestep )if noise_timestep and noise_timestep >0 else 0 ,
                    )

                    # Present. Image
                gen_imgs =self .diffusion_model .decode_first_stage (sampled_latents )
                gen_imgs =torch .clamp (gen_imgs ,-1. ,1. )
                gen_imgs =(gen_imgs +1.0 )/2 # Present. [0, 1]

                # Convert tonumpyCount
                for i in range (current_batch_size ):
                    gen_img_np =gen_imgs [i ].cpu ().numpy ().transpose ([1 ,2 ,0 ])
                    gen_img_np =np .clip (gen_img_np *255 ,0 ,255 ).astype (np .uint8 )
                    gen_img_np =gen_img_np [:,:,::-1 ]# RGB to BGR for OpenCV
                    generated_images .append (gen_img_np )

        return generated_images 

    def optimize_and_generate (
    self ,
    smiles :str ,
    target_ratio :Optional [float ]=None ,
    num_images :int =5 ,
    ddim_steps :int =50 ,
    eta :float =1.0 ,
    diffusion_batch_size :int =1 ,
    output_dir :str ="generated_molecules",
    noise_timestep :int =20 ,
    use_pure_noise :bool =False ,
    noise_strength :float =1.0 ,
    num_original_images :int =10 
    )->dict :
        """ andGenerate Args: smiles: InputSMILES target_ratio: Objective；used inModelavailableConditionavailableNone num_images: GenerateImageCount ddim_steps: DDIMSampleStepCount eta: DDIMSampleCount diffusion_batch_size: ModelBatch Size output_dir: Output Directory noise_timestep: NoiseIt is. .. Step use_pure_noise: availableYesUseNoise noise_strength: Noise num_original_images: GenerateEssaysImageIt is. .. Count，available0Generate Returns: OrganisationOutcomeInformationIt is. .. """
        print (f"Processing molecule: {smiles}")
        ratio_label =f"{target_ratio}"if target_ratio is not None else "unconditioned"
        print (f"Target ratio: {ratio_label}")
        print (f"Noise settings: timestep={noise_timestep}, strength={noise_strength}, pure_noise={use_pure_noise}")

        # 1. GetEssays
        print ("Getting original molecular representation. .. ")
        original_rep =self .get_molecular_representation (smiles )

        # 2. Essays
        print ("Optimizing molecular representation. .. ")
        optimized_rep =self .optimize_representation (smiles ,target_ratio ,target_property =self .target_property )

        original_property_pred =self ._predict_properties (original_rep )
        optimized_property_pred =self ._predict_properties (optimized_rep )
        property_delta =None 
        if original_property_pred and optimized_property_pred :
            property_delta ={}
            for prop ,stats in optimized_property_pred .items ():
                base_stats =original_property_pred .get (prop )
                if not base_stats :
                    continue 
                if 'value'in stats and 'value'in base_stats :
                    property_delta [f"{prop}_value"]=stats ['value']-base_stats ['value']
                if 'prob'in stats and 'prob'in base_stats :
                    property_delta [f"{prop}_prob"]=stats ['prob']-base_stats ['prob']

                    # 2.1 RecordsEssays（For ratio=1.0 It is time. Wait. Inspection）
        try :
            cos_sim =F .cosine_similarity (original_rep .unsqueeze (0 ),optimized_rep .unsqueeze (0 ),dim =-1 ).item ()
            l2_dist =torch .norm (optimized_rep -original_rep ,p =2 ).item ()
            print (f"Rep similarity: cosine={cos_sim: .4f}, L2={l2_dist: .4f}")
        except Exception as e :
            print (f"Warning: failed to compute rep similarity ({e})")

            # 3. GenerateUseEssaysImage（count）
        original_rep_images =[]
        if num_original_images >0 :
            print (f"Generating {num_original_images} images from original representation. .. ")
            original_rep_images =self .generate_images_from_representation (
            original_rep ,smiles ,num_original_images ,ddim_steps ,eta ,diffusion_batch_size ,
            noise_timestep ,use_pure_noise ,noise_strength 
            )
        else :
            print ("Skipping original representation image generation (num_original_images=0)")

            # 4. GenerateUseEssaysImage
        print (f"Generating {num_images} optimized molecular images. .. ")
        generated_images =self .generate_images_from_representation (
        optimized_rep ,smiles ,num_images ,ddim_steps ,eta ,diffusion_batch_size ,
        noise_timestep ,use_pure_noise ,noise_strength 
        )

        # 5. Save ImageandSMILESInformation
        os .makedirs (output_dir ,exist_ok =True )
        saved_paths =[]
        original_rep_paths =[]

        # SaveOriginal MolecularImage（RDKitGenerate）
        original_img =self .mol_to_image (smiles )
        original_path =os .path .join (output_dir ,f"original_{smiles. replace('/', '_')}.png")
        original_img .save (original_path )

        # SaveEssaysGenerateImage
        for i ,img in enumerate (original_rep_images ):
            img_path =os .path .join (output_dir ,f"original_rep_{smiles. replace('/', '_')}_img_{i+1}.png")
            cv2 .imwrite (img_path ,img )
            original_rep_paths .append (img_path )

            # SaveEssaysGenerateImage
        ratio_suffix =f"ratio_{target_ratio}"if target_ratio is not None else "uncond"
        for i ,img in enumerate (generated_images ):
            img_path =os .path .join (output_dir ,f"optimized_{smiles. replace('/', '_')}_{ratio_suffix}_img_{i+1}.png")
            cv2 .imwrite (img_path ,img )
            saved_paths .append (img_path )

            # SaveSMILESInformationPresent. CSVDocumentation
        csv_path =os .path .join (output_dir ,"source_smiles.csv")
        smiles_df =pd .DataFrame ({'source_smiles':[smiles ]})
        if os .path .exists (csv_path ):
        # If file, Data
            existing_df =pd .read_csv (csv_path )
            smiles_df =pd .concat ([existing_df ,smiles_df ],ignore_index =True )
        smiles_df .to_csv (csv_path ,index =False )

        result ={
        'original_smiles':smiles ,
        'target_ratio':target_ratio ,
        'original_image_path':original_path ,
        'original_rep_image_paths':original_rep_paths ,
        'generated_image_paths':saved_paths ,
        'num_original_rep_generated':len (original_rep_images ),
        'num_optimized_generated':len (generated_images ),
        'original_representation_shape':original_rep .shape ,
        'optimized_representation_shape':optimized_rep .shape ,
        'original_property_predictions':original_property_pred ,
        'optimized_property_predictions':optimized_property_pred ,
        'property_prediction_delta':property_delta ,
        }

        print (f"Generated {len(original_rep_images)} images from original representation")
        print (f"Generated {len(generated_images)} images from optimized representation")
        if original_property_pred :
            print ("Predicted properties (original representation): ")
            for prop ,stats in original_property_pred .items ():
                cfg =self .property_task_configs .get (prop ,{})
                if 'value'in stats :
                    print (f" - {prop} ({cfg. get('type', 'regression')}): {stats['value']: .4f}")
                elif 'prob'in stats :
                    print (f" - {prop} ({cfg. get('type', 'classification')}): prob={stats['prob']: .4f}, label={stats['label']: .0f}")
        if optimized_property_pred :
            print ("Predicted properties (optimized representation): ")
            for prop ,stats in optimized_property_pred .items ():
                cfg =self .property_task_configs .get (prop ,{})
                if 'value'in stats :
                    print (f" - {prop} ({cfg. get('type', 'regression')}): {stats['value']: .4f}")
                elif 'prob'in stats :
                    print (f" - {prop} ({cfg. get('type', 'classification')}): prob={stats['prob']: .4f}, label={stats['label']: .0f}")
        if property_delta :
            print ("Predicted property deltas (optimized - original): ")
            for key ,value in property_delta .items ():
                print (f" - {key}: {value: +. 4f}")
        print (f"All images saved to {output_dir}")
        return result 


def parse_args ():
    parser =argparse .ArgumentParser (description ='andImageGenerate')
    parser .add_argument ('--optimizer_model_path',type =str ,required =True ,
    help ='Model Path')
    parser .add_argument ('--rep_dim',type =int ,default =None ,
    help ='EssaysDimensions；If not provided，WillFromTrainingconfig. jsonRead，YesDefault512')
    parser .add_argument ('--hidden_dim',type =int ,default =None ,
    help ='LayerDimensions；If not provided，WillFromTrainingconfig. jsonRead，YesDefault512')
    parser .add_argument ('--num_blocks',type =int ,default =None ,
    help ='Convert BlocksCount；If not provided，WillFromTrainingconfig. jsonRead，YesDefault3')
    parser .add_argument ('--diffusion_config',type =str ,
    default ='config/ldm/dis_optmization.yaml',
    help ='ModelDocumentationPath')
    parser .add_argument ('--diffusion_model_path',type =str ,required =True ,
    help ='ModelWeightPath')
    parser .add_argument ('--encoder_path',type =str ,
    default ='checkpoints/pretrained_enc_ckpts/CGIP/CGIP.pth',
    help ='CGIPEncoderPath')
    parser .add_argument ('--property_predictor_checkpoint',type =str ,default =None ,
    help ='MultitaskProjectionsInspection，For')
    parser .add_argument ('--property_predictor_device',type =str ,default =None ,
    help ='ProjectionsRunEquipment（Defaultand --device Same）')
    parser .add_argument ('--target_property',type =str ,default =None ,
    help ='Objective；Model')
    parser .add_argument ('--property_list',type =str ,default =None ,
    help ='Columns（Comma Separated）；If not providedFromTraining config.json Read')
    parser .add_argument ('--mask_span',type =int ,default =None ,
    help ='Every one. Dimensions；If not providedReadTraining')
    parser .add_argument ('--mask_offset',type =int ,default =None ,
    help ='StartDimensions；If not providedReadTraining')
    parser .add_argument ('--use_attention',action ='store_true',
    help ='availableYesUseAttention. ')
    parser .add_argument ('--use_residual',action ='store_true',
    help ='availableYesUseImpairment')
    parser .add_argument ('--smiles',type =str ,
    help ='InputIt is. .. SMILES（Single）')
    parser .add_argument ('--csv_path',type =str ,default ='data/inference_data/test.csv',
    help ='OrganisationSMILESIt is. .. CSVDocumentationPath')
    parser .add_argument ('--csv_path_smiles_column',type =str ,default ='smiles',help ='OrganisationSMILESIt is. .. CSVDocumentationPath')
    parser .add_argument ('--ratio',type =float ,default =None ,
    help ='Objective；If not providedModelavailableCondition，CLI')
    parser .add_argument ('--num_images',type =int ,default =5 ,
    help ='GenerateImageCount')
    parser .add_argument ('--ddim_steps',type =int ,default =50 ,
    help ='DDIMSampleStepCount')
    parser .add_argument ('--eta',type =float ,default =1.0 ,
    help ='DDIMSampleCount')
    parser .add_argument ('--output_dir',type =str ,default ='generated_molecules',
    help ='Output Directory')
    parser .add_argument ('--device',type =str ,default ='cuda',
    help ='Equipment')
    parser .add_argument ('--diffusion_batch_size',type =int ,default =1 ,
    help ='ModelIt is time. batch size')
    parser .add_argument ('--max_molecules_random',action ='store_true',
    help ='used in，FromCSVMediumSample --max_molecules A molecule. ，Yes --max_molecules One. ')

    # NoiseCount
    parser .add_argument ('--noise_timestep',type =int ,default =20 ,
    help ='NoiseIt is. .. Step，Bigger. Noise，The smaller, the closer you get to the original image. (0-1000)')
    parser .add_argument ('--use_pure_noise',action ='store_true',
    help ='UseNoiseAs a. .. Start，Not the original image. ')
    parser .add_argument ('--noise_strength',type =float ,default =1.0 ,
    help ='Noise (0.0-2.0)，ForStepNoise')
    parser .add_argument ('--num_original_images',type =int ,default =10 ,
    help ='GenerateEssaysImageIt is. .. Count，If. .. available0GenerateEssaysImage')
    parser .add_argument ('--max_molecules',type =int ,default =None ,
    help ='ProcessingIt is. .. Count，If. .. ProcessingCSVMediumIt is. .. ')
    parser .add_argument ('--use_ratio_condition',dest ='use_ratio_condition',action ='store_true',
    help ='Use it. ratioConditionInput（DefaultTraining）')
    parser .add_argument ('--no_ratio_condition',dest ='use_ratio_condition',action ='store_false',
    help ='Use it. ratioConditionInput')
    parser .add_argument ('--seed',type =int ,default =None ,
    help ='Random Feeds，Sample')
    parser .set_defaults (use_ratio_condition =None )

    return parser .parse_args ()


def main ():
    args =parse_args ()
    if args .seed is not None :
        set_global_seed (args .seed )
        print (f"Set global random seed to {args. seed}")

        # FromTrainingSaveLoadModel（If）
    rep_dim =args .rep_dim 
    hidden_dim =args .hidden_dim 
    num_blocks =args .num_blocks 
    cfg ={}
    if rep_dim is None or hidden_dim is None or num_blocks is None :
        try :
            train_dir =os .path .dirname (args .optimizer_model_path )
            cfg_path =os .path .join (train_dir ,'config.json')
            if os .path .exists (cfg_path ):
                with open (cfg_path ,'r')as f :
                    cfg =json .load (f )
                rep_dim =rep_dim if rep_dim is not None else int (cfg .get ('rep_dim',512 ))
                hidden_dim =hidden_dim if hidden_dim is not None else int (cfg .get ('hidden_dim',512 ))
                num_blocks =num_blocks if num_blocks is not None else int (cfg .get ('num_blocks',3 ))
                print (f"Loaded arch params from {cfg_path}: rep_dim={rep_dim}, hidden_dim={hidden_dim}, num_blocks={num_blocks}")
            else :
            # Default
                rep_dim =512 if rep_dim is None else rep_dim 
                hidden_dim =512 if hidden_dim is None else hidden_dim 
                num_blocks =3 if num_blocks is None else num_blocks 
                print (f"No config.json found alongside checkpoint. Using defaults: rep_dim={rep_dim}, hidden_dim={hidden_dim}, num_blocks={num_blocks}")
        except Exception as e :
        # Back toDefault
            rep_dim =512 if rep_dim is None else rep_dim 
            hidden_dim =512 if hidden_dim is None else hidden_dim 
            num_blocks =3 if num_blocks is None else num_blocks 
            print (f"Warning: failed to load arch params from config.json ({e}). Using defaults: rep_dim={rep_dim}, hidden_dim={hidden_dim}, num_blocks={num_blocks}")

    use_ratio_condition =args .use_ratio_condition 
    if use_ratio_condition is None :
        use_ratio_condition =bool (cfg .get ('use_ratio_condition',False ))if cfg else False 
    print (f"Ratio conditioning enabled: {use_ratio_condition}")
    if use_ratio_condition and args .ratio is None :
        raise ValueError ("Ratio conditioning is enabled but no --ratio value was provided. ")
    if not use_ratio_condition and args .ratio is not None :
        print ("Warning: --ratio provided but ratio conditioning is disabled; value will be ignored. ")

    if args .property_list :
        property_names =[p .strip ()for p in args .property_list .split (', ')if p .strip ()]
    else :
        property_names =cfg .get ('property_names',[])if cfg else []
    target_property =args .target_property or (cfg .get ('target_property')if cfg else None )
    if target_property is None and property_names :
        target_property =property_names [0 ]
    if not property_names and target_property :
        property_names =[target_property ]
    if not property_names :
        raise ValueError ("Columns，Please. --property_list orPromise. TrainingOrganisation property_names")
    mask_span =args .mask_span if args .mask_span is not None else cfg .get ('mask_span',20 )if cfg else 20 
    mask_offset =args .mask_offset if args .mask_offset is not None else cfg .get ('mask_offset',0 )if cfg else 0 

    # CreateOptimizer
    optimizer =MolecularOptimizerWithGeneration (
    optimizer_model_path =args .optimizer_model_path ,
    diffusion_model_config =args .diffusion_config ,
    diffusion_model_path =args .diffusion_model_path ,
    encoder_path =args .encoder_path ,
    device =args .device ,
    use_attention =args .use_attention ,
    use_residual =args .use_residual ,
    rep_dim =rep_dim ,
    hidden_dim =hidden_dim ,
    num_blocks =num_blocks ,
    use_ratio_condition =use_ratio_condition ,
    property_names =property_names ,
    target_property =target_property ,
    property_masks_span =mask_span ,
    property_mask_offset =mask_offset ,
    property_predictor_checkpoint =args .property_predictor_checkpoint ,
    property_predictor_device =args .property_predictor_device ,
    )

    # ProcessingSMILEScolumns
    if args .smiles :
    # If SingleSMILES, ProcessingOne.
        smiles_list =[args .smiles ]
        print (f"Processing single SMILES: {args. smiles}")
    else :
    # FromCSVDocumentationReadSMILES
        print (f"Reading SMILES from CSV file: {args. csv_path}")
        df =pd .read_csv (args .csv_path )
        if args .csv_path_smiles_column not in df .columns :
            raise ValueError (f"CSVDocumentationMediumNo Found'{args. csv_path_smiles_column}'Columns")
        smiles_list =df [args .csv_path_smiles_column ].tolist ()

        # ProcessingCount
        if args .max_molecules is not None and args .max_molecules >0 :
            limit =min (args .max_molecules ,len (smiles_list ))
            if args .max_molecules_random :
                smiles_list =random .sample (smiles_list ,limit )
                print (f"Randomly selected {len(smiles_list)} SMILES to process")
            else :
                smiles_list =smiles_list [:limit ]
                print (f"Limited to first {len(smiles_list)} SMILES to process")
        else :
            print (f"Found {len(smiles_list)} SMILES to process")

            # ProcessingEvery one. SMILES
    for idx ,smiles in enumerate (smiles_list ):
        print (f"\n{'='*60}")
        print (f"Processing [{idx+1}/{len(smiles_list)}]: {smiles}")
        print (f"{'='*60}")
        start_time =time .perf_counter ()
        try :
        # Every one. Create
            safe_smiles =smiles .replace ('/','_').replace ('\\','_')
            mol_output_dir =os .path .join (args .output_dir ,f"mol_{idx+1}_{safe_smiles}")

            # andGenerate
            result =optimizer .optimize_and_generate (
            smiles =smiles ,
            target_ratio =args .ratio if use_ratio_condition else None ,
            num_images =args .num_images ,
            ddim_steps =args .ddim_steps ,
            eta =args .eta ,
            diffusion_batch_size =args .diffusion_batch_size ,
            output_dir =mol_output_dir ,
            noise_timestep =args .noise_timestep ,
            use_pure_noise =args .use_pure_noise ,
            noise_strength =args .noise_strength ,
            num_original_images =args .num_original_images 
            )

            # Outcome
            print ("\n"+"="*50 )
            print (f"GENERATION COMPLETE for molecule {idx+1}!")
            print ("="*50 )
            print (f"Original SMILES: {result['original_smiles']}")
            print (f"Target ratio: {result['target_ratio']}")
            print (f"Generated {result['num_original_rep_generated']} images from original representation")
            print (f"Generated {result['num_optimized_generated']} optimized images")
            print (f"Original image (RDKit) saved to: {result['original_image_path']}")
            print ("Original representation generated images saved to: ")
            for path in result ['original_rep_image_paths']:
                print (f" - {path}")
            print ("Optimized representation generated images saved to: ")
            for path in result ['generated_image_paths']:
                print (f" - {path}")
            print ("="*50 )

        except Exception as e :
            print (f"\nError processing SMILES {smiles}: {str(e)}")
            print (f"Skipping to next molecule. .. \n")
        finally :
            elapsed =time .perf_counter ()-start_time 
            print (f"Processing time for molecule {idx+1}: {elapsed: .2f} seconds")

    print (f"\n{'='*60}")
    print ("ALL MOLECULES PROCESSED!")
    print (f"{'='*60}")


if __name__ =='__main__':
    main ()
