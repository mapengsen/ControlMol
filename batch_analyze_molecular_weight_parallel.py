import os 
import subprocess 
import pandas as pd 
from rdkit import Chem 
import glob 
import tempfile 
import argparse 
import multiprocessing 
from multiprocessing import Pool 
import time 
import sys 
import warnings 
warnings .filterwarnings ('ignore')

try :
    from tqdm import tqdm 
    TQDM_AVAILABLE =True 
except ImportError :
    TQDM_AVAILABLE =False 

VERBOSE =False 
MODEL_CACHE ={
'model':None ,
'path':None ,
'device':None ,
'available':False 
}


def vprint (*args ,**kwargs ):
    """According toVERBOSESwitch whether to print logs"""
    if VERBOSE :
        print (*args ,**kwargs )


def select_device_and_log ():
    """Selectioncudaorcpu，and print the device currently in use"""
    device =torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')
    device_name =None 
    if device .type =='cuda':
        try :
            device_name =torch .cuda .get_device_name (torch .cuda .current_device ())
        except Exception :
            device_name =None 
    message =f"[Process {os. getpid()}] Use of equipment: {device. type}"
    if device_name :
        message +=f" ({device_name})"
    print (message )
    return device 


def init_worker (verbose_flag ,model_path ):
    """Every one. ProcessLoadModeland"""
    global VERBOSE ,MODEL_CACHE 
    VERBOSE =verbose_flag 
    if MOLSCRIBE_AVAILABLE and model_path :
        try :
            device =select_device_and_log ()
            MODEL_CACHE ['model']=MolScribe (model_path ,device )
            MODEL_CACHE ['path']=model_path 
            MODEL_CACHE ['device']=device 
            MODEL_CACHE ['available']=True 
            vprint (f"[Process {os. getpid()}] Model preloaded successfully，Equipment: {device}")
        except Exception as exc :
            MODEL_CACHE ['model']=None 
            MODEL_CACHE ['available']=False 
            if VERBOSE :
                print (f"[Process {os. getpid()}] Synchronising folder failed: %s: %s: {exc}")

                # Addevaluation/MolScribeTo Path, _Other Organisermolscribe
sys .path .append (os .path .join (os .path .dirname (__file__ ),'evaluation','MolScribe'))

try :
    import torch 
    from molscribe import MolScribe 
    MOLSCRIBE_AVAILABLE =True 
except ImportError as e :
    MOLSCRIBE_AVAILABLE =False 

def get_image_paths (image_folder ):
    """ Path to fetch all image files under folder Args: image_folder (str): Image Folder Path Returns: list: Processing"""
    image_paths =[]
    for root ,_ ,files in os .walk (image_folder ):
        for file in files :
            if file .lower ().endswith (('.png','.jpg','.jpeg','.bmp','.tiff','.gif')):
                image_paths .append (os .path .join (root ,file ))
    return image_paths 

def predict_images_with_molscribe_direct (image_folder ,model ,batch_size =128 ):
    """ Direct UseMolScribeModel predicts all molecular structures in the image folder Args: image_folder (str): Image Folder Path model: LoadedMolScribeModel examples batch_size (int): Batch Size Returns: list: List with projected results，♪ Every element is ♪dictFormat """
    if model is None :
        vprint (f"[Process {os. getpid()}] Error: Model not loaded")
        return []

        # Fetch All Image Paths in Folder
    image_paths =get_image_paths (image_folder )

    if not image_paths :
        vprint (f"[Process {os. getpid()}] No image files found in {image_folder}")
        return []

    vprint (f"[Process {os. getpid()}] Found {len(image_paths)} image files")

    results =[]

    # Process images in batches
    for i in range (0 ,len (image_paths ),batch_size ):
        batch_images =image_paths [i :i +batch_size ]

        try :
        # Using model predictions
            output =model .predict_image_files (batch_images ,return_atoms_bonds =False ,return_confidence =True )

            # Result of processing
            for img_path ,result in zip (batch_images ,output ):
                confidence =result .get ('confidence',0.0 )

                # Filtering Low Results
                if confidence >=0.0 :
                    image_name =os .path .basename (img_path )
                    smiles =result .get ('smiles','')

                    # Validate predicted SMILES
                    if smiles and is_valid_smiles (smiles ):
                        results .append ({
                        'imageName':image_name ,
                        'smiles':smiles ,
                        'confidence':confidence 
                        })

            vprint (f"[Process {os. getpid()}] Batch processing {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")

        except Exception as e :
            vprint (f"[Process {os. getpid()}] Batch processing failed: {e}")
            continue 

    vprint (f"[Process {os. getpid()}] Successfully predicted {len(results)} molecular structures")
    return results 

def is_valid_smiles (smiles ):
    """Return whether a SMILES string is valid."""
    try :
        if '.'in smiles or '*'in smiles :
            return False 
        mol =Chem .MolFromSmiles (smiles )
        return mol is not None 
    except :
        return False 

def predict_images_with_molscribe (image_folder ,model_path ):
    """Run MolScribe on all images in a folder and return a prediction DataFrame."""
    # Create temporary output file
    temp_csv =tempfile .NamedTemporaryFile (suffix ='.csv',delete =False )
    temp_csv_path =temp_csv .name 
    temp_csv .close ()

    try :
    # Build Command, Direct UsePythonEnvironment
        cmd =[
        sys .executable ,
        "evaluation/MolScribe/predict_images.py",
        "--model_path",model_path ,
        "--image_folder",image_folder ,
        "--output_csv",temp_csv_path 
        ]

        vprint (f"[Process {os. getpid()}] Processing folder: {image_folder}")

        # Execute Command
        result =subprocess .run (cmd ,capture_output =True ,text =True ,cwd =os .getcwd ())

        if result .returncode ==0 :
        # Read ResultsCSV
            if os .path .exists (temp_csv_path )and os .path .getsize (temp_csv_path )>0 :
                df =pd .read_csv (temp_csv_path )
                vprint (f"[Process {os. getpid()}] Successfully predicted {len(df)} molecular structures")
                return df 
            else :
                vprint (f"[Process {os. getpid()}] Prediction output file is empty or missing")
                return None 
        else :
            print (f"[Process {os. getpid()}] Prediction command failed:")
            print (f"stderr: {result. stderr}")
            return None 

    except Exception as e :
        print (f"[Process {os. getpid()}] CLI prediction error: {e}")
        return None 
    finally :
    # Clear temporary files
        if os .path .exists (temp_csv_path ):
            os .unlink (temp_csv_path )

def get_all_subfolders_with_images (input_output_folder ):
    """Return all immediate subfolders under the input root that contain image files."""
    folders_with_images =[]

    # Walk through the first subfolder under the root directory
    if not os .path .exists (input_output_folder ):
        print (f"Error: folder does not exist: {input_output_folder}")
        return folders_with_images 

    for item in os .listdir (input_output_folder ):
        item_path =os .path .join (input_output_folder ,item )
        if os .path .isdir (item_path ):
        # Check whether the subfolder contains images
            image_files =get_image_paths (item_path )# Reuse get_image_paths to detect image-containing folders
            if image_files :
                folders_with_images .append (item_path )

    return sorted (folders_with_images )

def process_single_folder (folder_info ):
    """Process one folder and return SMILES recognition statistics."""
    global MODEL_CACHE 

    folder_path ,model_path ,batch_size =folder_info 
    folder_name =os .path .basename (folder_path )

    start_time =time .time ()
    vprint (f"[Process {os. getpid()}] Start processing folder: {folder_name}")

    # Reuse a process-local model cache when possible
    model =None 
    using_preloaded_model =False 
    if MODEL_CACHE ['available']and MODEL_CACHE ['model']is not None and MODEL_CACHE ['path']==model_path :
        model =MODEL_CACHE ['model']
        using_preloaded_model =True 
    elif MOLSCRIBE_AVAILABLE :
        try :
            vprint (f"[Process {os. getpid()}] Loading MolScribe model...")
            device =select_device_and_log ()
            model =MolScribe (model_path ,device )
            MODEL_CACHE ['model']=model 
            MODEL_CACHE ['path']=model_path 
            MODEL_CACHE ['device']=device 
            MODEL_CACHE ['available']=True 
            using_preloaded_model =True 
            vprint (f"[Process {os. getpid()}] Model loaded successfully on device: {device}")
        except Exception as e :
            MODEL_CACHE ['model']=None 
            MODEL_CACHE ['available']=False 
            vprint (f"[Process {os. getpid()}] ModelLoadFailed: {e}")
            vprint (f"[Process {os. getpid()}] Back to Command Line Mode")

            # Find all subfolders containing images
    subfolders =[]
    for root ,dirs ,files in os .walk (folder_path ):
    # Find subfolders that actually contain images
        has_images =any (file .lower ().endswith (('.png','.jpg','.jpeg','.bmp','.tiff','.gif'))
        for file in files )
        if has_images :
            subfolders .append (root )

    if not subfolders :
        print (f"[Process {os. getpid()}] No image subfolders found in {folder_name}")
        return {
        'folder_name':folder_name ,
        'success':False ,
        'error':'No image subfolders found',
        'processing_time':time .time ()-start_time 
        }

    vprint (f"[Process {os. getpid()}] Found {len(subfolders)} subfolders containing images")

    # Aggregate SMILES statistics
    total_valid_smiles =0 
    total_predictions =0 
    folder_details ={
    'folder_name':folder_name ,
    'subfolder_count':len (subfolders ),
    'subfolder_details':[],
    'total_valid_smiles':0 ,
    'total_predictions':0 
    }

    for i ,subfolder in enumerate (subfolders ):
        vprint (f"[Process {os. getpid()}] Process subfolder {i+1}/{len(subfolders)}: {os. path. relpath(subfolder, folder_path)}")

        # Readsource_smiles
        source_smiles_value =None 
        source_smiles_path =os .path .join (subfolder ,'source_smiles.csv')
        if os .path .exists (source_smiles_path ):
            try :
                df_source =pd .read_csv (source_smiles_path )
                if 'source_smiles'in df_source .columns :
                    source_column =df_source ['source_smiles'].dropna ().astype (str ).str .strip ()
                    if not source_column .empty :
                        source_smiles_value =source_column .iloc [0 ]
            except Exception as exc :
                vprint (f"[Process {os. getpid()}] Failed to read {source_smiles_path}: {exc}")

                # UseLoadModelorBack to
        if model is not None :
        # UseLoadModel（Efficient Mode）
            prediction_results =predict_images_with_molscribe_direct (subfolder ,model ,batch_size )
        else :
        # Back to Command Line Mode
            vprint (f"[Process {os. getpid()}] Using CLI fallback...")
            df_pred =predict_images_with_molscribe (subfolder ,model_path )
            # Convert the CLI output to a consistent schema
            prediction_results =[]
            if df_pred is not None :
                for _ ,row in df_pred .iterrows ():
                    smiles_value =row .get ('smiles','')if 'smiles'in row else row .get ('SMILES','')
                    prediction_results .append ({
                    'smiles':smiles_value ,
                    'confidence':row .get ('confidence',0.0 ),
                    'imageName':row .get ('imageName',row .get ('image_name',''))
                    })

        subfolder_detail ={
        'path':os .path .relpath (subfolder ,folder_path ),
        'image_count':len (prediction_results ),
        'valid_smiles_count':0 ,
        'smiles_data':[],# Store per-image SMILES predictions
        'source_smiles':source_smiles_value 
        }

        if prediction_results :
            valid_count =0 
            for result in prediction_results :
                smiles =result .get ('smiles','')
                confidence =result .get ('confidence',0.0 )
                image_name =result .get ('imageName','')

                if smiles and smiles .strip ():
                    smiles_clean =smiles .strip ()
                    if is_valid_smiles (smiles_clean ):
                        valid_count +=1 
                        smiles_info ={
                        'image_name':image_name ,
                        'smiles':smiles_clean ,
                        'confidence':confidence ,
                        'source_smiles':source_smiles_value 
                        }
                        subfolder_detail ['smiles_data'].append (smiles_info )

            subfolder_detail ['valid_smiles_count']=valid_count 
            vprint (f"[Process {os. getpid()}] - Prediction count: {len(prediction_results)}")
            vprint (f"[Process {os. getpid()}] - Valid SMILES count: {valid_count}")
            total_predictions +=len (prediction_results )
            total_valid_smiles +=valid_count 
        else :
            vprint (f"[Process {os. getpid()}] - Folder prediction failed or returned no results")

        folder_details ['subfolder_details'].append (subfolder_detail )
        folder_details ['total_valid_smiles']=total_valid_smiles 
        folder_details ['total_predictions']=total_predictions 

        # General statistical information
    processing_time =time .time ()-start_time 
    folder_details ['total_valid_smiles']=total_valid_smiles 
    folder_details ['total_predictions']=total_predictions 

    result ={
    'folder_name':folder_name ,
    'success':total_valid_smiles >0 ,
    'total_valid_smiles':total_valid_smiles ,
    'total_predictions':total_predictions ,
    'subfolder_count':len (subfolders ),
    'processing_time':processing_time ,
    'details':folder_details ,
    'model_preloaded':using_preloaded_model 
    }

    if total_valid_smiles ==0 :
        result ['error']='No valid SMILES predictions found'

    vprint (f"[Process {os. getpid()}] 📊 {folder_name} Overall statistics: ")
    vprint (f"[Process {os. getpid()}] - Folder count: {len(subfolders)}")
    vprint (f"[Process {os. getpid()}] - Prediction count: {total_predictions}")
    vprint (f"[Process {os. getpid()}] - Valid SMILES count: {total_valid_smiles}")
    vprint (f"[Process {os. getpid()}] - Processing Time: {processing_time: .1f}sec")
    vprint (f"[Process {os. getpid()}] - Model loading mode: {'preloaded' if using_preloaded_model else 'on-demand'}")

    if total_valid_smiles ==0 :
        vprint (f"[Process {os. getpid()}] ❌ {folder_name} produced no valid SMILES predictions")

    return result 

def recognize_smiles_parallel (input_output_folder ,model_path ,num_processes =4 ,batch_size =128 ):
    """Process all folders containing images in parallel and collect SMILES recognition results."""
    # Automatically fetch all folders containing images
    target_folders =get_all_subfolders_with_images (input_output_folder )

    if not target_folders :
        print (f"No folders containing images were found under {input_output_folder}")
        return {},[]

    vprint (f"Found {len(target_folders)} folders containing images:")
    for folder in target_folders :
        vprint (f" - {os. path. basename(folder)}")

        # Build per-folder processing tasks
    tasks =[(folder_path ,model_path ,batch_size )for folder_path in target_folders ]

    # Bound worker count by folders and CPU capacity
    actual_processes =min (num_processes ,len (target_folders ),multiprocessing .cpu_count ())

    vprint (f"\nUse {actual_processes} worker processes for {len(target_folders)} folders")
    vprint (f"MolScribe availability: {'available' if MOLSCRIBE_AVAILABLE else 'unavailable (will use CLI fallback)'}")
    vprint (f"Batch Size: {batch_size}")

    # Start parallel processing
    start_time =time .time ()

    try :
        with Pool (processes =actual_processes ,
        initializer =init_worker ,
        initargs =(VERBOSE ,model_path if MOLSCRIBE_AVAILABLE else None ))as pool :
            if TQDM_AVAILABLE :
                process_results =[]
                with tqdm (total =len (tasks ),desc ="Processing",unit ="folder",ncols =80 )as pbar :
                    for result in pool .imap_unordered (process_single_folder ,tasks ):
                        process_results .append (result )
                        pbar .update ()
            else :
                print ("Hint: tqdm is not installed, progress bar disabled")
                # Parallel assignments
                process_results =pool .map (process_single_folder ,tasks )
    except Exception as e :
        print (f"Parallel processing failed: {e}")
        return {},[]

    total_time =time .time ()-start_time 
    print (f"\nAll folders processed. Total time: {total_time: .1f}sec")

    # ModelLoad
    preloaded_count =sum (1 for r in process_results if r .get ('model_preloaded',False ))
    vprint (f"Workers with preloaded model: {preloaded_count}/{len(process_results)}")

    # Collate Results
    results ={}
    detailed_results =[]

    for process_result in process_results :
        folder_name =process_result ['folder_name']
        folder_data ={
        'total_valid_smiles':process_result .get ('total_valid_smiles',0 ),
        'total_predictions':process_result .get ('total_predictions',0 ),
        'subfolder_count':process_result .get ('subfolder_count',0 ),
        'processing_time':process_result .get ('processing_time',0.0 ),
        'model_preloaded':process_result .get ('model_preloaded',False )
        }
        if not process_result .get ('success',False ):
            folder_data ['error']=process_result .get ('error','Process Failed')
        results [folder_name ]=folder_data 

        if 'details'in process_result :
            detailed_results .append (process_result ['details'])

    return results ,detailed_results 

def save_results_to_files (results ,detailed_results ,output_dir ="."):
    """Save summary and detailed SMILES recognition results to disk."""
    summary_file =os .path .join (output_dir ,"smiles_analysis_summary.txt")
    with open (summary_file ,'w',encoding ='utf-8')as f :
        f .write ("SMILES Recognition Summary\n")
        f .write ("="*50 +"\n\n")

        total_processing_time =0 
        for folder_name ,data in results .items ():
            f .write (f"Folder: {folder_name}\n")
            f .write (f" Valid SMILES count: {data['total_valid_smiles']}\n")
            f .write (f" Prediction count: {data['total_predictions']}\n")
            f .write (f" Subfolder count: {data['subfolder_count']}\n")
            f .write (f" Processing Time: {data['processing_time']: .1f}sec\n")
            if 'error'in data :
                f .write (f" Error: {data['error']}\n")
            f .write ("\n")
            total_processing_time +=data ['processing_time']

        f .write (f"Total processing time (sum of workers): {total_processing_time: .1f}sec\n")

    csv_rows =[]
    for folder_detail in detailed_results :
        folder_name =folder_detail ['folder_name']
        folder_total_valid =folder_detail .get ('total_valid_smiles',0 )
        folder_total_predictions =folder_detail .get ('total_predictions',0 )

        for subfolder_detail in folder_detail ['subfolder_details']:
            csv_rows .append ({
            'main_folder':folder_name ,
            'subfolder_path':subfolder_detail ['path'],
            'prediction_count':subfolder_detail ['image_count'],
            'valid_smiles_count':subfolder_detail ['valid_smiles_count'],
            'folder_total_valid_smiles':folder_total_valid ,
            'folder_total_predictions':folder_total_predictions ,
            'source_smiles':subfolder_detail .get ('source_smiles')or ''
            })

    if csv_rows :
        df_results =pd .DataFrame (csv_rows )
        csv_file =os .path .join (output_dir ,"smiles_subfolder_summary.csv")
        df_results .to_csv (csv_file ,index =False ,encoding ='utf-8')
        print (f"Folder summary saved to: {csv_file}")

    smiles_rows =[]
    for folder_detail in detailed_results :
        folder_name =folder_detail ['folder_name']

        for subfolder_detail in folder_detail ['subfolder_details']:
            subfolder_path =subfolder_detail ['path']

            if 'smiles_data'in subfolder_detail :
                for smiles_info in subfolder_detail ['smiles_data']:
                    smiles_rows .append ({
                    'main_folder':folder_name ,
                    'subfolder_path':subfolder_path ,
                    'image_name':smiles_info .get ('image_name',''),
                    'smiles':smiles_info .get ('smiles',''),
                    'confidence':smiles_info .get ('confidence',None ),
                    'source_smiles':smiles_info .get ('source_smiles')or ''
                    })

    if smiles_rows :
        df_smiles =pd .DataFrame (smiles_rows )
        smiles_csv_file =os .path .join (output_dir ,"smiles_predictions_detailed.csv")
        df_smiles .to_csv (smiles_csv_file ,index =False ,encoding ='utf-8')
        print (f"Detailed SMILES results saved to: {smiles_csv_file}")

    print (f"Summary results saved to: {summary_file}")

def main ():
    parser =argparse .ArgumentParser (description='Batch process molecular images and summarize valid SMILES predictions')
    parser .add_argument ('--input_output_folder',type =str ,default ='results',
    help='Input root directory (default: results)')
    parser .add_argument ('--model_path',type =str ,
    default ='evaluation/MolScribe/ckpt_from_molscribe/swin_base_char_aux_1m680k.pth',
    help='MolScribe checkpoint path')
    parser .add_argument ('--output_dir',type =str ,default =None ,
    help='Output directory (default: same as input_output_folder)')
    parser .add_argument ('--num_processes',type =int ,default =4 ,
    help='Number of worker processes (default: 4)')
    parser .add_argument ('--batch_size',type =int ,default =128 ,
    help='MolScribe batch size (default: 128)')
    parser .add_argument ('--verbose',action ='store_true',
    help='Print detailed processing logs')

    args =parser .parse_args ()

    global VERBOSE 
    VERBOSE =args .verbose 

    # Validate the input folder
    if not os .path .exists (args .input_output_folder ):
        print (f"Error: input folder does not exist: {args. input_output_folder}")
        return 

    if args .output_dir is None :
        args .output_dir =args .input_output_folder 

        # Make sure. Output Directory, Avoiding back-to-back document errors
    os .makedirs (args .output_dir ,exist_ok =True )

    # Validate the MolScribe checkpoint path
    if not os .path .exists (args .model_path ):
        print (f"Error: model file does not exist: {args. model_path}")
        return 

    vprint (f"Start batch molecular image recognition...")
    vprint (f"Input folder: {args. input_output_folder}")
    vprint (f"Model Path: {args. model_path}")
    vprint (f"Output Directory: {args. output_dir}")
    vprint (f"Worker process count: {args. num_processes}")
    vprint (f"Batch Size: {args. batch_size}")

    # Run recognition and summarize valid SMILES
    results ,detailed_results =recognize_smiles_parallel (
    args .input_output_folder ,
    args .model_path ,
    args .num_processes ,
    args .batch_size 
    )

    if not results :
        print ("No valid prediction data found. Exiting.")
        return 

        # Show final result banner
    print (f"\n{'='*90}")

    # Save Results to File
    save_results_to_files (results ,detailed_results ,args .output_dir )

    print (f"\nProcessing completed!")

if __name__ =="__main__":
    multiprocessing .set_start_method ('spawn',force =True )# Setup startup method, Avoiding some platform problems
    main ()
