import math 
import sys 
from typing import Iterable 
import os 
import time 

import torch 

import util .misc as misc 
import util .lr_sched as lr_sched 

import cv2 

import numpy as np 
import shutil 
from rdkit import Chem 
import pandas as pd 
import torch 
import warnings 
warnings .filterwarnings ('ignore')
import os 
import csv 
import pandas as pd 




def train_one_epoch (model :torch .nn .Module ,
data_loader :Iterable ,optimizer :torch .optim .Optimizer ,
device :torch .device ,epoch :int ,loss_scaler ,
log_writer =None ,
args =None ):
    model .train (True )
    metric_logger =misc .MetricLogger (delimiter =" ")
    metric_logger .add_meter ('lr',misc .SmoothedValue (window_size =1 ,fmt ='{value: .6f}'))
    header ='Epoch: [{}]'.format (epoch )
    print_freq =20 

    accum_iter =args .accum_iter 

    optimizer .zero_grad ()

    if log_writer is not None :
        print ('log_dir: {}'.format (log_writer .log_dir ))

    for data_iter_step ,data_batch in enumerate (metric_logger .log_every (data_loader ,print_freq ,header )):
    # Single Task
        if len (data_batch )==5 :
            images_pos ,labels_pos ,images_neg ,labels_neg ,task_ids =data_batch 
            images_pos =images_pos .to (device ,non_blocking =True )
            images_neg =images_neg .to (device ,non_blocking =True )
            labels_pos =labels_pos .to (device ,non_blocking =True )
            labels_neg =labels_neg .to (device ,non_blocking =True )
            task_ids =task_ids .to (device ,non_blocking =True )

            # Multitask
        else :
            images_pos1 ,labels_pos1 ,images_pos2 ,labels_pos2 ,images_neg ,labels_neg ,task_ids =data_batch 

            # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step %accum_iter ==0 and args .cosine_lr :
            lr_sched .adjust_learning_rate (optimizer ,data_iter_step /len (data_loader )+epoch ,args )

            # TODO: YesUse
            # images = images * 2 - 1 # image to [-1, 1] to be compatible with LDM

        batch ={'images_pos':images_pos ,'images_neg':images_neg ,
        'labels_pos':labels_pos ,'labels_neg':labels_neg ,
        'task_ids':task_ids }

        loss ,loss_dict =model (x =None ,c =None ,batch =batch ,**vars (args ))

        loss_value =loss .item ()

        if not math .isfinite (loss_value ):
            print ("Loss is {}, stopping training".format (loss_value ))
            sys .exit (1 )

        loss /=accum_iter 
        loss_scaler (loss ,optimizer ,parameters =model .parameters (),update_grad =(data_iter_step +1 )%accum_iter ==0 )
        if (data_iter_step +1 )%accum_iter ==0 :
            optimizer .zero_grad ()

        torch .cuda .synchronize ()
        # Printedloss
        metric_logger .update (loss_total =loss_value )
        metric_logger .update (loss_mse =loss_dict ['train/loss_simple'])
        # metric_logger. update(loss_rep=loss_dict['train/loss_rep'])

        lr =optimizer .param_groups [0 ]["lr"]
        metric_logger .update (lr =lr )

        loss_value_reduce =misc .all_reduce_mean (loss_value )
        loss_value_mse =misc .all_reduce_mean (loss_dict ['train/loss_simple'])
        # loss_value_rep = misc. all_reduce_mean(loss_dict['train/loss_rep'])
        # tensorboardInside. loss
        if log_writer is not None and (data_iter_step +1 )%accum_iter ==0 :
            """ We use epoch_1000x as the x-axis in tensorboard. This calibrates different curves when batch size changes. """
            epoch_1000x =int ((data_iter_step /len (data_loader )+epoch )*1000 )
            log_writer .add_scalar ('train_loss',loss_value_reduce ,epoch_1000x )
            log_writer .add_scalar ('train_mse',loss_value_mse ,epoch_1000x )
            # log_writer. add_scalar('train_rep', loss_value_rep, epoch_1000x)
            log_writer .add_scalar ('lr',lr ,epoch_1000x )

            # gather the stats from all processes
    metric_logger .synchronize_between_processes ()
    print ("Averaged stats: ",metric_logger )
    return {k :meter .global_avg for k ,meter in metric_logger .meters .items ()}


def gen_img (model ,args ,epoch ,batch_size =16 ,log_writer =None ):
    model .eval ()
    num_steps =args .num_images //(batch_size *misc .get_world_size ())+1 
    print ("Total need generation is: ",args .num_images )
    print ("Total need generation iteration: ",num_steps )
    save_folder =os .path .join (args .output_dir ,"steps{}-eta{}".format (args .ldm_steps ,args .eta ))
    if misc .get_rank ()==0 :
        if not os .path .exists (save_folder ):
            os .makedirs (save_folder )

    for i in range (num_steps ):
        print ("Generation step {}/{}".format (i ,num_steps ))

        with torch .no_grad ():
            gen_images_batch =model (x =None ,c =None ,gen_img =True )
        gen_images_batch =gen_images_batch .detach ().cpu ()

        # save img
        if misc .get_rank ()==0 :
            for b_id in range (gen_images_batch .size (0 )):
                if i *gen_images_batch .size (0 )+b_id >=args .num_images :
                    break 
                gen_img =np .clip (gen_images_batch [b_id ].numpy ().transpose ([1 ,2 ,0 ])*255 ,0 ,255 )
                gen_img =gen_img .astype (np .uint8 )[:,:,::-1 ]

                # Secure access to file names
                try :
                # Usedata_batch[1]As a SMILES
                    if isinstance (data_batch [1 ],list )and b_id <len (data_batch [1 ]):
                        file_name =f"{data_batch[1][b_id]}_{epoch}"
                    else :
                    # If data_batch[1]columnsorIndexScope, UseIndexAs a file
                        file_name =f"img_{b_id}_{epoch}"
                except (IndexError ,TypeError ):
                    print ("Error")
                    # If Error, UseIndexAs a file
                    file_name =f"img_{b_id}_{epoch}"

                save_img_name_path =os .path .join (save_folder ,f"{file_name}.png")
                cv2 .imwrite (save_img_name_path ,gen_img )







def get_image_paths (image_folder ):
# Use os. walk Folder, Fetch All File Paths
    image_paths =[]
    for root ,_ ,files in os .walk (image_folder ):
        for file in files :
            if file .lower ().endswith (('.png','.jpg','.jpeg')):
                image_paths .append (os .path .join (root ,file ))
    return image_paths 


    # one CountInspection SMILES Validity
def is_valid_smiles (smiles ):
    try :
        mol =Chem .MolFromSmiles (smiles )
        return mol is not None 
    except :
        return False 


        # Global variable, ForInitializationModel
model_MolScribe =None 
admet_model =None 
batch_size_MolScribe =64 
ADMETModel =None 
DEFAULT_DRUGBANK_PATH =None 
DEFAULT_MODELS_DIR =None 
MolScribe =None 

def initialize_models ():
    """availableused inGPUUp InitializationMolScribeandADMETModel"""
    global model_MolScribe ,admet_model 
    global ADMETModel ,DEFAULT_DRUGBANK_PATH ,DEFAULT_MODELS_DIR ,MolScribe 

    if MolScribe is None :
        from evaluation .MolScribe .molscribe import MolScribe as _MolScribe 
        MolScribe =_MolScribe 

    if ADMETModel is None :
        from admet_ai import ADMETModel as _ADMETModel 
        from admet_ai .constants import (
        DEFAULT_DRUGBANK_PATH as _DEFAULT_DRUGBANK_PATH ,
        DEFAULT_MODELS_DIR as _DEFAULT_MODELS_DIR ,
        )

        ADMETModel =_ADMETModel 
        DEFAULT_DRUGBANK_PATH =_DEFAULT_DRUGBANK_PATH 
        DEFAULT_MODELS_DIR =_DEFAULT_MODELS_DIR 

        # GetProcessGPUEquipment
    current_device =torch .cuda .current_device ()
    device =torch .device (f'cuda: {current_device}')

    if model_MolScribe is None :
        try :
            model_path ='evaluation/MolScribe/ckpt_from_molscribe/swin_base_char_aux_1m680k.pth'
            model_MolScribe =MolScribe (model_path ,device )
            print (f"Rank {misc. get_rank()}: MolScribeModelavailable {device} Up Initialization")
        except Exception as e :
            print (f"Rank {misc. get_rank()}: MolScribeInitialisation failed: {e}")
            # If Failed, UseCPU
            device_cpu =torch .device ('cpu')
            model_MolScribe =MolScribe (model_path ,device_cpu )
            print (f"Rank {misc. get_rank()}: MolScribeDowngrade toCPURun")

    if admet_model is None :
        models_dir =DEFAULT_MODELS_DIR 
        drugbank_path =DEFAULT_DRUGBANK_PATH 
        include_physchem =True 
        no_cache_molecules =True 

        # BuildADMETModel
        admet_model =ADMETModel (
        models_dir =models_dir ,
        include_physchem =include_physchem ,
        drugbank_path =drugbank_path ,
        atc_code =None ,
        num_workers =0 ,
        cache_molecules =not no_cache_molecules ,
        )
        print (f"Rank {misc. get_rank()}: ADMETModelInitializationSuccess")



        # Definition of classification rules 【WillReturnsClassesStoragecsv】
classification_rules ={
'molecular_weight':lambda x :(x <=500 ).astype (int ),
'logP':lambda x :(x <=5 ).astype (int ),
'hydrogen_bond_acceptors':lambda x :(x <=5 ).astype (int ),
'hydrogen_bond_donors':lambda x :(x <=10 ).astype (int ),
'QED':lambda x :(x >=0.5 ).astype (int ),
'tpsa':lambda x :(x <=140 ).astype (int ),
'RoB':lambda x :(x <=10 ).astype (int ),
'SA':lambda x :(x <=5 ).astype (int ),
'AMES':lambda x :(x >=0.5 ).astype (int ),
'BBB_Martins':lambda x :(x >=0.5 ).astype (int ),
'Bioavailability_Ma':lambda x :(x >=0.5 ).astype (int ),
'CYP1A2_Veith':lambda x :(x >=0.5 ).astype (int ),
'CYP2C19_Veith':lambda x :(x >=0.5 ).astype (int ),
'CYP2C9_Substrate_CarbonMangels':lambda x :(x >=0.5 ).astype (int ),
'CYP2C9_Veith':lambda x :(x >=0.5 ).astype (int ),
'CYP2D6_Substrate_CarbonMangels':lambda x :(x >=0.5 ).astype (int ),
'CYP2D6_Veith':lambda x :(x >=0.5 ).astype (int ),
'CYP3A4_Substrate_CarbonMangels':lambda x :(x >=0.5 ).astype (int ),
'CYP3A4_Veith':lambda x :(x >=0.5 ).astype (int ),
'Carcinogens_Lagunin':lambda x :(x >=0.5 ).astype (int ),
'ClinTox':lambda x :(x >=0.5 ).astype (int ),
'DILI':lambda x :(x >=0.5 ).astype (int ),
'HIA_Hou':lambda x :(x >=0.5 ).astype (int ),
'NR-AR-LBD':lambda x :(x >=0.5 ).astype (int ),
'NR-AR':lambda x :(x >=0.5 ).astype (int ),
'NR-AhR':lambda x :(x >=0.5 ).astype (int ),
'NR-Aromatase':lambda x :(x >=0.5 ).astype (int ),
'NR-ER-LBD':lambda x :(x >=0.5 ).astype (int ),
'NR-ER':lambda x :(x >=0.5 ).astype (int ),
'NR-PPAR-gamma':lambda x :(x >=0.5 ).astype (int ),
'PAMPA_NCATS':lambda x :(x >=0.5 ).astype (int ),
'Pgp_Broccatelli':lambda x :(x >=0.5 ).astype (int ),
'SR-ARE':lambda x :(x >=0.5 ).astype (int ),
'SR-ATAD5':lambda x :(x >=0.5 ).astype (int ),
'SR-HSE':lambda x :(x >=0.5 ).astype (int ),
'SR-MMP':lambda x :(x >=0.5 ).astype (int ),
'SR-p53':lambda x :(x >=0.5 ).astype (int ),
'Skin_Reaction':lambda x :(x >=0.5 ).astype (int ),
'hERG':lambda x :(x >=0.5 ).astype (int ),
'Clearance_Hepatocyte_AZ':lambda x :(x <5.1 ).astype (int ),
'Clearance_Microsome_AZ':lambda x :(x <15 ).astype (int ),
'HydrationFreeEnergy_FreeSolv':lambda x :(x <=-3 ).astype (int ),
'LD50_Zhu':lambda x :(x <1.56 ).astype (int ),
'Lipophilicity_AstraZeneca':lambda x :(x <=3 ).astype (int ),
'PPBR_AZ':lambda x :(x >=90 ).astype (int ),
'Solubility_AqSolDB':lambda x :(x >=-2 ).astype (int ),
'VDss_Lombardo':lambda x :(x >2 ).astype (int ),
}


def gen_recognition_img_disentanglement_rl (model ,args ,data_batch ,data_iter_step =None ,epoch =None ):
    """VersionIt is. .. ImageGenerateandCount，BackProjectionsOutcomeandlogProbability"""
    model .train ()

    # InspectionMolScribeandADMETModelYesInitialization
    global model_MolScribe ,admet_model 
    if model_MolScribe is None or admet_model is None :
        raise RuntimeError ("MolScribeandADMETModelInitialization！availableMediumCallinitialize_models()Count")

    save_folder =args .output_dir 
    # Every one. ProcessCreateOne. OnlyFolder, Avoiding conflict
    rank =misc .get_rank ()
    unique_temp_folder =os .path .join (save_folder ,f'temp_images_rank_{rank}')
    temp_image_folder_iter =os .path .join (save_folder ,'temp_images_iter')# This can be shared.

    os .makedirs (unique_temp_folder ,exist_ok =True )
    os .makedirs (temp_image_folder_iter ,exist_ok =True )

    # Generate images and get themlogProbability
    gen_images_batch ,log_prob ,rep =model .module .gen_imgs_disentanglement_trainable_with_log_prob (data_batch ,args )if hasattr (model ,'module')else model .gen_imgs_disentanglement_trainable_with_log_prob (data_batch ,args )
    gen_images_batch_cpu =gen_images_batch .detach ().cpu ()

    # Save Image
    save_img_name_path_list =[]

    # AddStep, Avoid allGPUWriting documents at the same time
    if torch .distributed .is_initialized ():
    # According torankScatterI/OOperation
        world_size =misc .get_world_size ()
        # Every one. rankWait for a little while.
        time .sleep (rank *0.1 )

        # Every one. gpuSave Image
    for b_id in range (gen_images_batch_cpu .size (0 )):
        gen_img =np .clip (gen_images_batch_cpu [b_id ].numpy ().transpose ([1 ,2 ,0 ])*255 ,0 ,255 )
        gen_img =gen_img .astype (np .uint8 )[:,:,::-1 ]

        if isinstance (data_batch ,(list ,tuple ))and len (data_batch )>1 :
            if isinstance (data_batch [1 ],list )and b_id <len (data_batch [1 ]):
                file_name =f"{data_batch[1][b_id]}_{epoch}_rank{misc. get_rank()}_{b_id}"
            else :
                file_name =f"img_{b_id}_{epoch}_rank{misc. get_rank()}"
        else :
            file_name =f"img_{b_id}_{epoch}_rank{misc. get_rank()}"

        save_img_name_path =os .path .join (unique_temp_folder ,f"{file_name}.png")
        save_img_name_path_list .append (save_img_name_path )
        cv2 .imwrite (save_img_name_path ,gen_img )

        if data_iter_step %args .save_temp_image_iter ==0 :
            cv2 .imwrite (os .path .join (temp_image_folder_iter ,f"{file_name}.png"),gen_img )


            # ReleaseCPUImage Memory
    del gen_images_batch_cpu 
    torch .cuda .empty_cache ()

    # Identification of Images（andCountSameLogical）
    image_paths =save_img_name_path_list 
    valid_results =[]
    un_valid_results =[]

    # Press batch_size Handle pictures
    for i in range (0 ,len (image_paths ),batch_size_MolScribe ):
        batch_images =image_paths [i :i +batch_size_MolScribe ]
        output =model_MolScribe .predict_image_files (batch_images ,return_atoms_bonds =False ,return_confidence =True )

        # Result of processingandFilterSMILES, Keep the original index at the same time
        for j ,(img_path ,result )in enumerate (zip (batch_images ,output )):
            original_index =i +j # Compute Original Index
            confidence =round (result .get ('confidence'),3 )
            smiles =result .get ('smiles')

            # Filter Conditions: ValiditySMILES + Non-empty + Single Molecular（No Point Separator）
            if is_valid_smiles (smiles )and smiles !=''and '.'not in smiles :
                image_name =img_path 
                valid_results .append ({
                'original_index':original_index ,
                'imageName':image_name ,
                'smiles':smiles ,
                'confidence':confidence 
                })
            else :
                image_name =os .path .splitext (img_path )[0 ]+'_Error_smiles'+os .path .splitext (img_path )[1 ]
                un_valid_results .append ({
                'original_index':original_index ,
                'imageName':image_name ,
                'smiles':smiles ,
                'confidence':confidence 
                })

        del output 
        if i %3 ==0 :
            torch .cuda .empty_cache ()

            # ADMETprediction
    final_df =None 
    final_valid_df =None 
    final_unvalid_df =None 

    if valid_results :
        smiles_list =[result ['smiles']for result in valid_results ]
        preds =admet_model .predict (smiles =smiles_list )

        base_data =pd .DataFrame ({
        'original_index':[result ['original_index']for result in valid_results ],
        'Drug':smiles_list ,
        'imageName':[result ['imageName']for result in valid_results ],
        'confidence':[result ['confidence']for result in valid_results ]
        }).reset_index (drop =True )

        if isinstance (preds ,pd .DataFrame ):
            preds =preds .reset_index (drop =True )

        min_len =min (len (base_data ),len (preds ))
        base_data =base_data .iloc [:min_len ]
        preds =preds .iloc [:min_len ]

        data_with_preds =pd .concat ([base_data ,preds ],axis =1 ,ignore_index =False )

        for column ,rule in classification_rules .items ():
            if column in data_with_preds .columns :
                data_with_preds [column ]=rule (data_with_preds [column ])

        final_valid_df =data_with_preds 
        del preds ,base_data ,data_with_preds 

    if un_valid_results :
        smiles_list =[result ['smiles']for result in un_valid_results ]
        base_data =pd .DataFrame ({
        'original_index':[result ['original_index']for result in un_valid_results ],
        'Drug':smiles_list ,
        'imageName':[result ['imageName']for result in un_valid_results ],
        'confidence':[result ['confidence']for result in un_valid_results ]
        }).reset_index (drop =True )

        prop_columns =[col for col in classification_rules .keys ()]

        for col in prop_columns :
            base_data [col ]=99999 

        final_unvalid_df =base_data 
        del base_data ,smiles_list 


        # Willfinal_unvalid_dfandfinal_valid_dfPressProcessingImage, Go back to the final one. final_df
    if final_valid_df is not None and final_unvalid_df is not None :
    # Merge twoDataFrame
        final_df =pd .concat ([final_valid_df ,final_unvalid_df ],axis =0 ,ignore_index =True )
    elif final_valid_df is not None :
        final_df =final_valid_df 
    elif final_unvalid_df is not None :
        final_df =final_unvalid_df 
    else :
        raise RuntimeError ("No GenerateValidityorIt is. .. Outcome")

        # PressIndex, Keep the original image processing order
    if final_df is not None and len (final_df )>0 :
        final_df =final_df .sort_values ('original_index').reset_index (drop =True )
        # Remove an Auxiliaryoriginal_indexcolumns
        final_df =final_df .drop ('original_index',axis =1 )

    del valid_results ,un_valid_results 
    import gc 
    gc .collect ()
    torch .cuda .empty_cache ()

    # Clear temporary image files
    try :
        for img_path in save_img_name_path_list :
            if os .path .exists (img_path ):
                os .remove (img_path )
    except Exception as e :
        print (f"Clear temporary filesFailed: {e}")

    return final_df ,log_prob ,rep 
















