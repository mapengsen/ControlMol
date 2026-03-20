import os 
import sys 
import argparse 
import json 
from typing import Dict ,List ,Set ,Tuple 

import torch 
import pandas as pd 
from tqdm import tqdm 
from rdkit import Chem 
from rdkit .Chem import Draw 
from PIL import Image 
import torchvision .transforms as transforms 
from torch .utils .data import Dataset ,DataLoader 

# Add the project root so the pretrained encoder can be imported
sys .path .append (os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
import pretrained_enc .models_pretrained_enc as models_pretrained_enc 


def scale_to_range (x :torch .Tensor )->torch .Tensor :
# Map image tensors from [0, 1] to [-1, 1] to match train.py
    return x *2.0 -1.0 


def normalize_representation (rep :torch .Tensor )->torch .Tensor :
# Match the per-sample z-score normalization used in train.py
    rep_std =torch .std (rep ,dim =1 ,keepdim =True )
    rep_mean =torch .mean (rep ,dim =1 ,keepdim =True )
    rep_std =torch .clamp (rep_std ,min =1e-8 )
    return (rep -rep_mean )/rep_std 


def parallel_batch_mol_to_images (smiles_batch :List [str ]):
    """Convert a batch of SMILES strings to RDKit-rendered PIL images."""
    images =[]
    valid_indices =[]
    failed_count =0 

    for i ,smiles in enumerate (smiles_batch ):
        try :
            mol =Chem .MolFromSmiles (smiles )
            if mol is None :
                failed_count +=1 
                continue 
            img =Draw .MolToImage (mol ,size =(256 ,256 ),imageType ='png')
            images .append (img )
            valid_indices .append (i )
        except Exception :
            failed_count +=1 

    return images ,valid_indices ,failed_count 


    # UseProcess,


class EncoderWrapper :
    """Wrapper that reproduces the encoder preprocessing used in train.py."""

    def __init__ (self ,encoder :torch .nn .Module ):
        self .encoder =encoder 
        self .transform =transforms .Compose ([
        transforms .ToTensor (),
        transforms .Lambda (scale_to_range )
        ])

    @torch .no_grad ()
    def encode_images (self ,images :List [Image .Image ],batch_size :int =128 )->List [torch .Tensor ]:
        device =next (self .encoder .parameters ()).device 
        reps :List [torch .Tensor ]=[]

        for i in range (0 ,len (images ),batch_size ):
            batch_imgs =images [i :i +batch_size ]
            tensors =[self .transform (img )for img in batch_imgs ]
            if not tensors :
                continue 
            x =torch .stack (tensors ).to (device )

            # Apply the same ImageNet normalization used in train.py
            mean =torch .tensor ([0.485 ,0.456 ,0.406 ],device =device ).view (1 ,3 ,1 ,1 )
            std =torch .tensor ([0.229 ,0.224 ,0.225 ],device =device ).view (1 ,3 ,1 ,1 )

            # [-1, 1] -> [0, 1]
            if x .min ()>=-1.1 and x .max ()<=1.1 :
                x_norm =(x +1 )/2 
            else :
                x_norm =x 

            x_norm =(x_norm -mean )/std 
            x_norm =torch .nn .functional .interpolate (x_norm ,224 ,mode ='bicubic',align_corners =False )

            batch_reps =self .encoder (x_norm )# [B, 512, 1, 1]
            batch_reps =batch_reps .squeeze (-1 ).squeeze (-1 )# [B, 512]
            batch_reps =normalize_representation (batch_reps )# z-score per sample
            reps .extend (batch_reps .cpu ())

        return reps 


def iter_unique_smiles (csv_path :str ,smiles_columns :List [str ],chunksize :int =500_000 )->Set [str ]:
    """Read a CSV file and collect unique SMILES from the selected columns."""
    if not smiles_columns :
        raise ValueError ("smiles_columns is empty")
    smiles_columns =[col .strip ()for col in smiles_columns if col .strip ()]
    if not smiles_columns :
        raise ValueError ("smiles_columns is empty")

    unique :Set [str ]=set ()
    for chunk in pd .read_csv (csv_path ,usecols =smiles_columns ,chunksize =chunksize ):
        for col in smiles_columns :
            if col not in chunk .columns :
                continue 
            unique .update (chunk [col ].dropna ().astype (str ).tolist ())
    return unique 


class SmilesImageDataset (Dataset ):
    """Dataset that converts SMILES to images and returns tensors compatible with train.py preprocessing."""
    def __init__ (self ,smiles_list :List [str ]):
        self .smiles_list =smiles_list 
        self .transform =transforms .Compose ([
        transforms .ToTensor (),
        transforms .Lambda (scale_to_range )
        ])

    def __len__ (self )->int :
        return len (self .smiles_list )

    def __getitem__ (self ,idx :int )->Tuple [str ,torch .Tensor ]:
        smiles =self .smiles_list [idx ]
        try :
            mol =Chem .MolFromSmiles (smiles )
            if mol is None :
            # 
                img =Image .new ('RGB',(256 ,256 ),color ='white')
            else :
                img =Draw .MolToImage (mol ,size =(256 ,256 ),imageType ='png')
        except Exception :
            img =Image .new ('RGB',(256 ,256 ),color ='white')

        tensor =self .transform (img )
        return smiles ,tensor 


def smiles_collate (batch :List [Tuple [str ,torch .Tensor ]])->Tuple [List [str ],torch .Tensor ]:
    smiles =[b [0 ]for b in batch ]
    tensors =torch .stack ([b [1 ]for b in batch ],dim =0 )
    return smiles ,tensors 


def compute_and_save_cache (
csv_path :str ,
output_path :str ,
encoder_path :str ,
encode_batch_size :int =128 ,
image_batch_size :int =500 ,
chunksize :int =500_000 ,
num_workers :int =0 ,
prefetch_factor :int =2 ,
smiles_columns :List [str ]=None ,
):
    os .makedirs (os .path .dirname (output_path )or '.',exist_ok =True )

    device =torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')
    print (f"Using device: {device}")

    # Load the same encoder used in train.py
    encoder =models_pretrained_enc .__dict__ ['CGIP_image_model']()
    if os .path .exists (encoder_path ):
        encoder =models_pretrained_enc .load_pretrained_CGIP_image_ckpt (encoder ,encoder_path )
        print ("Loaded CGIP pretrained weights")
    else :
        print (f"Warning: encoder checkpoint not found at {encoder_path}; using the default pretrained weights")
    encoder .to (device )
    encoder .eval ()

    smiles_columns =smiles_columns or ['start','final']
    smiles_columns =[col .strip ()for col in smiles_columns if col .strip ()]
    print (f"Collect unique SMILES from columns: {', '. join(smiles_columns)}")
    try :
        unique_smiles =iter_unique_smiles (csv_path ,smiles_columns =smiles_columns ,chunksize =chunksize )
    except ValueError as exc :
        raise ValueError (f"Failed to read the CSV with columns {smiles_columns}: {exc}")from exc 
    unique_smiles =list (unique_smiles )
    print (f"Unique SMILES count: {len(unique_smiles): ,}")

    smiles_to_rep :Dict [str ,torch .Tensor ]={}
    failed =0 # Count SMILES that failed during rendering or encoding

    if num_workers and num_workers >0 :
        print (f"Use DataLoader-based RDKit rendering: workers={num_workers}, prefetch_factor={prefetch_factor}")
        dataset =SmilesImageDataset (unique_smiles )
        loader =DataLoader (
        dataset ,
        batch_size =encode_batch_size ,
        shuffle =False ,
        num_workers =num_workers ,
        collate_fn =smiles_collate ,
        pin_memory =True ,
        prefetch_factor =prefetch_factor ,
        persistent_workers =True ,
        )

        mean =torch .tensor ([0.485 ,0.456 ,0.406 ],device =device ).view (1 ,3 ,1 ,1 )
        std =torch .tensor ([0.229 ,0.224 ,0.225 ],device =device ).view (1 ,3 ,1 ,1 )

        pbar =tqdm (total =len (unique_smiles ),desc ="Encoding",unit ="mol")
        with torch .no_grad ():
            for smiles_batch ,cpu_tensors in loader :
            # Move batched tensors to the target device
                x =cpu_tensors .to (device ,non_blocking =True )

                # Convert back to [0, 1], then apply ImageNet normalization and bicubic resizing
                if x .min ()>=-1.1 and x .max ()<=1.1 :
                    x =(x +1 )/2 
                x =(x -mean )/std 
                x =torch .nn .functional .interpolate (x ,224 ,mode ='bicubic',align_corners =False )

                reps =encoder (x ).squeeze (-1 ).squeeze (-1 )
                reps =normalize_representation (reps ).cpu ()

                for smi ,rep in zip (smiles_batch ,reps ):
                    smiles_to_rep [smi ]=rep .clone ()

                pbar .update (len (smiles_batch ))
        pbar .close ()
    else :
    # Render images and encode them in the main process
        print ("Generate images and encode them in the main process")
        encoder_wrapper =EncoderWrapper (encoder )
        failed =0 

        # Split SMILES into blocks for rendering and encoding
        block =image_batch_size 
        pbar =tqdm (total =len (unique_smiles ),desc ="Encoding",unit ="mol")
        for start in range (0 ,len (unique_smiles ),block ):
            smiles_block =unique_smiles [start :start +block ]

            imgs ,valid_idx ,fail_cnt =parallel_batch_mol_to_images (smiles_block )
            all_images =imgs 
            all_indices =valid_idx 
            failed +=fail_cnt 

            # Encode rendered images
            reps =encoder_wrapper .encode_images (all_images ,batch_size =encode_batch_size )

            # Store encoded representations in a dict[str, Tensor]
            for local_i ,rep in zip (all_indices ,reps ):
                smiles =smiles_block [local_i ]
                smiles_to_rep [smiles ]=rep .clone ()# CPU tensor

                # Update progress bar
            pbar .update (len (smiles_block ))
        pbar .close ()

    print (f"Encoding finished: success={len(smiles_to_rep):,}, failed={failed:,}")

    # Save
    print (f"Saved cache to: {output_path}")
    torch .save (smiles_to_rep ,output_path )

    meta ={
    'total_unique':len (unique_smiles ),
    'cached':len (smiles_to_rep ),
    'failed':failed ,
    'encoder_path':encoder_path ,
    'device':str (device ),
    'smiles_columns':smiles_columns ,
    }
    meta_path =os .path .splitext (output_path )[0 ]+'. meta.json'
    with open (meta_path ,'w',encoding ='utf-8')as f :
        json .dump (meta ,f ,ensure_ascii =False ,indent =2 )
    print (f"Saved metadata to: {meta_path}")


def parse_args ():
    p =argparse .ArgumentParser (description='Precompute a SMILES-to-representation cache for train.py')
    p .add_argument ('--csv_path',type =str ,required =True ,help='CSV path containing SMILES columns such as start and final')
    p .add_argument ('--output_path',type =str ,required =True ,help='Output path for the SMILES-to-representation cache (.pt)')
    p .add_argument ('--encoder_path',type =str ,default ='checkpoints/pretrained_enc_ckpts/CGIP/CGIP.pth')
    p .add_argument ('--encode_batch_size',type =int ,default =128 ,help='Batch size used when encoding images')
    p .add_argument ('--image_batch_size',type =int ,default =500 ,help='Number of SMILES rendered per block in main-process mode')
    p .add_argument ('--chunksize',type =int ,default =500_000 ,help='Chunk size used when reading the CSV')
    p .add_argument ('--num_workers',type =int ,default =0 ,help='Number of DataLoader workers for RDKit rendering (0 disables workers)')
    p .add_argument ('--prefetch_factor',type =int ,default =2 ,help='Prefetch factor for DataLoader workers')
    p .add_argument ('--smiles_columns',type =str ,nargs ='+',default =['start','final'],help='SMILES columns to include when building the cache')
    return p .parse_args ()


if __name__ =='__main__':
    args =parse_args ()
    compute_and_save_cache (
    csv_path =args .csv_path ,
    output_path =args .output_path ,
    encoder_path =args .encoder_path ,
    encode_batch_size =args .encode_batch_size ,
    image_batch_size =args .image_batch_size ,
    chunksize =args .chunksize ,
    num_workers =args .num_workers ,
    prefetch_factor =args .prefetch_factor ,
    smiles_columns =args .smiles_columns ,
    )
