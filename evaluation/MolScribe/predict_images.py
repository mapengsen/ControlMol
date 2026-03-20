import argparse 
import torch 
from molscribe import MolScribe 
import warnings 
warnings .filterwarnings ('ignore')
import os 
import csv 
import pandas as pd 
from rdkit import Chem 

"""
pip install OpenNMT-py==2.2.0
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip uninstall opencv-contrib-python-headless
pip3 install opencv-contrib-python==4.5.5.62
pip install albumentations@git+https: //github. com/albumentations-team/albumentations@37e714fd2e326f6f88778e425f98c2de8c8d5372
or
pip install albumentations==1.1.0
pip install timm==0.4.12 conda activate ldm python evaluation/MolScribe/predict_images.py --model_path evaluation/MolScribe/ckpt_from_molscribe/swin_base_char_aux_1m680k.pth \
--image_folder data/genertaion_image/mw_100_200 \
--output_csv data/image2smiles/image2smiles.csv """



def get_image_paths (image_folder ):
# Use os. walk Folder, Fetch All File Paths
    image_paths =[]
    for root ,_ ,files in os .walk (image_folder ):
        for file in files :
            if file .lower ().endswith (('.png','.jpg','.jpeg')):
                image_paths .append (os .path .join (root ,file ))
    return image_paths 

if __name__ =="__main__":
    parser =argparse .ArgumentParser ()
    parser .add_argument ('--model_path',type =str ,required =True ,help ="Path to the model")
    parser .add_argument ('--image_folder',type =str ,required =True ,help ="Path to the folder containing images")
    parser .add_argument ("--batch_size",default =128 ,type =int ,help ="Batch size for processing images")
    parser .add_argument ("--output_csv",type =str ,default ="output.csv",help ="Output CSV file path")
    args =parser .parse_args ()

    device =torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')
    model =MolScribe (args .model_path ,device )

    # GetFolderMediumPath
    image_paths =get_image_paths (args .image_folder )
    print (f"Found {len(image_paths)} images in the folder. ")

    with open (args .output_csv ,mode ='w',newline ='')as file :
        writer =csv .writer (file )
        writer .writerow (['imageName','canonical_smiles','confidence'])# Write to table header

        # Press batch_size Handle pictures
        for i in range (0 ,len (image_paths ),args .batch_size ):
            batch_images =image_paths [i :i +args .batch_size ]
            output =model .predict_image_files (batch_images ,return_atoms_bonds =False ,return_confidence =True )

            # WillOutcome CSV file
            for img_path ,result in zip (batch_images ,output ):
                confidence =round (result .get ('confidence','N/A'),3 )
                # # Delete less confidence than0.1Outcome
                if confidence <=0.0 :
                    continue 
                else :
                    image_name =os .path .basename (img_path )# Get
                    smiles =result .get ('smiles','N/A')
                    writer .writerow ([image_name ,smiles ,confidence ])

            print (f"Processed batch {(i // args. batch_size + 1) * args. batch_size}")

    print (f"Results saved to {args. output_csv}")


    # one CountInspection SMILES Validity
    def is_valid_smiles (smiles ):
        try :
            mol =Chem .MolFromSmiles (smiles )
            return mol is not None 
        except :
            return False 


            # Read CSV file
    df =pd .read_csv (args .output_csv )
    # Can not delete folder: %s: No such folder '.' or '*'
    df =df [~df ['canonical_smiles'].str .contains (r'\. |[*]',na =False )]
    # StandardizationSMILES
    df =df [df ['canonical_smiles'].apply (is_valid_smiles )]
    # SaveOutcomePresent. file
    df .to_csv (args .output_csv ,index =False )
















