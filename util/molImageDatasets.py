# import os
# import pandas as pd
# from PIL import Image
# import torch
# from torch. utils. data import Dataset, DataLoader
# from torchvision import transforms
# import numpy as np
#
#
# class MultiTaskMolecularDataset(Dataset):
# def __init__(self, annotations_file, img_dir, target_task_names, transform=None, image_suffix='.png'):
# """
# Args:
# annotations_file (string): CSVSummaryDocumentationPath.
# img_dir (string): ImageFolderPath.
# target_task_names (list): DatasetExampleObjectiveTaskscolumns.
# transform (callable, optional): ForOptional.
# image_suffix (string, optional): image file.
# """
# self. img_dir = img_dir
# self. transform = transform
# self. image_suffix = image_suffix
# self. target_task_names = target_task_names if isinstance(target_task_names, list) else [target_task_names]
#
# # Build a task identifier
# self. task_id = '+'. join(self. target_task_names)
#
# # Load annotation CSV
# self. all_annotations = pd. read_csv(annotations_file)
#
# # Property columns excluding metadata
# basic_columns = ['Drug', 'compoundName', 'task', 'stereo_centers', 'Lipinski', 'Caco2_Wang', 'Half_Life_Obach']
# self. property_columns = [col for col in self. all_annotations. columns if col not in basic_columns]
#
# print(f"DetectedcolumnsCount: {len(self. property_columns)}")
# print(f"columns: {self. property_columns}")
#
# # 1. Filter rows for the requested tasks
# task_specific_data = self. all_annotations[self. all_annotations['task']. isin(self. target_task_names)]. copy()
#
# # 2. Collect compounds for the requested tasks
# if len(self. target_task_names) == 1:
# # Single Task
# self. valid_compounds = list(task_specific_data['compoundName']. unique())
# else:
# # Multitask: keep only compounds shared across tasks
# compound_sets = []
# for task in self. target_task_names:
# task_compounds = set(task_specific_data[task_specific_data['task'] == task]['compoundName']. unique())
# compound_sets. append(task_compounds)
#
# # Intersection
# self. valid_compounds = list(set. intersection(*compound_sets))
#
# # 3. Keep compounds that have image files
# existing_compounds = []
# for compound in self. valid_compounds:
# img_path = os. path. join(self. img_dir, compound + self. image_suffix)
# if os. path. exists(img_path):
# existing_compounds. append(compound)
# else:
# raise FileNotFoundError(f"image file: {img_path}")
#
# self. valid_compounds = existing_compounds
#
# # 4. Every one. ValidityCompoundsCreateNature Dictionary
# self. compound_properties = {}
# for compound in self. valid_compounds:
# # Load rows for the current compound
# compound_data = task_specific_data[task_specific_data['compoundName'] == compound]
#
# if len(compound_data) > 0:
# # If Data, （orAccording toOne. Logical）
# row = compound_data. iloc[0]
#
# # Build the property dictionary
# properties = {}
# for prop_col in self. property_columns:
# value = row[prop_col]
# # ProcessingNaNValue
# if pd. isna(value):
# properties[prop_col] = None
# else:
# properties[prop_col] = value
#
# self. compound_properties[compound] = properties
# else:
# print(f"Warning: Compounds {compound} No Data")
#
# print(f"Tasks {self.target_task_names}: valid compound count = {len(self. valid_compounds)}")
#
# def __len__(self):
# return len(self. valid_compounds)
#
# def _load_image(self, compoundName):
# """count, ForLoadSingleImage"""
# img_filename = compoundName + self. image_suffix
# img_path = os. path. join(self. img_dir, img_filename)
#
# try:
# image = Image. open(img_path). convert('RGB')
# if self. transform:
# image = self. transform(image)
# return image
# except FileNotFoundError:
# print(f"Warning: Present. image file {img_path}")
# return None
# except Exception as e:
# print(f"Warning: LoadImage {img_path} Time error: {e}")
# return None
# finally:
# # Ensure image resources are released
# try:
# if 'image' in locals() and hasattr(image, 'close'):
# image. close()
# except:
# pass
#
# def __getitem__(self, idx):
# if idx >= len(self. valid_compounds):
# raise IndexError(f"Index {idx} Scope, ValidityCompoundsCount {len(self. valid_compounds)}")
#
# compoundName = self. valid_compounds[idx]
# image = self. _load_image(compoundName)
# if image is None:
# raise ValueError(f"ImageLoadFailed: {compoundName}")
# properties = self. compound_properties. get(compoundName, {})
#
# # Back: Image, Compounds, Tasks, Nature Dictionary
# return image, compoundName, self. task_id, properties
#
#
# def collate_fn_skip_invalid(batch):
# """Batch processingDatacollateCount"""
# # Image, Compounds, TasksandNature Dictionary
# images = [item[0] for item in batch]
# compoundNames = [item[1] for item in batch]
# task_names = [item[2] for item in batch]
# properties_list = [item[3] for item in batch]
#
# # Stack images into a batch tensor
# images_batch = torch. stack(images)
#
# # Reorganize property dictionaries by property name
# # Build one batched dict entry for each property column
# if properties_list:
# # Get
# all_property_keys = set()
# for props in properties_list:
# if props is not None: # Make sure. None
# all_property_keys. update(props. keys())
#
# # Create the batched property dictionary
# batch_properties = {}
# for key in all_property_keys:
# batch_properties[key] = []
# for props in properties_list:
# if props is not None and key in props: # InspectionandYes
# batch_properties[key]. append(props. get(key, None))
# else:
# batch_properties[key]. append(None)
# else:
# batch_properties = {}
#
# return images_batch, compoundNames, task_names, batch_properties


import os 
import pandas as pd 
from PIL import Image 
import torch 
from torch .utils .data import Dataset ,DataLoader 
from torchvision import transforms 
import numpy as np 


class MultiTaskMolecularDataset (Dataset ):
    def __init__ (self ,annotations_file ,img_dir ,target_task_names ,transform =None ,image_suffix ='.png'):
        """Load a multitask molecular image dataset from CSV annotations and image files."""
        self .img_dir =img_dir 
        self .transform =transform 
        self .image_suffix =image_suffix 
        self .target_task_names =target_task_names if isinstance (target_task_names ,list )else [target_task_names ]

        # Build a task identifier
        self .task_id ='+'.join (self .target_task_names )

        # Load annotation CSV
        self .all_annotations =pd .read_csv (annotations_file )

        # Property columns excluding metadata
        basic_columns =['Drug','compoundName','task','stereo_centers','Lipinski','Caco2_Wang','Half_Life_Obach']
        self .property_columns =[col for col in self .all_annotations .columns if col not in basic_columns ]

        print (f"Detected property column count: {len(self. property_columns)}")
        print (f"Columns: {self. property_columns}")

        # 1. Filter rows for the requested tasks
        task_specific_data =self .all_annotations [self .all_annotations ['task'].isin (self .target_task_names )].copy ()

        # 2. Collect compounds for the requested tasks
        if len (self .target_task_names )==1 :
        # Single Task
            self .valid_compounds =list (task_specific_data ['compoundName'].unique ())
        else :
        # Multitask: keep only compounds shared across tasks
            compound_sets =[]
            for task in self .target_task_names :
                task_compounds =set (task_specific_data [task_specific_data ['task']==task ]['compoundName'].unique ())
                compound_sets .append (task_compounds )

                # Intersection
            self .valid_compounds =list (set .intersection (*compound_sets ))

            # 3. Keep compounds that have image files
        existing_compounds =[]
        for compound in self .valid_compounds :
            img_path =os .path .join (self .img_dir ,compound +self .image_suffix )
            if os .path .exists (img_path ):
                existing_compounds .append (compound )
            else :
                raise FileNotFoundError (f"Image file not found: {img_path}")

        self .valid_compounds =existing_compounds 

        # 4. Build property dictionaries and cache SMILES for valid compounds
        self .compound_properties ={}
        self .compound_smiles ={}# Cache the SMILES string for each compound
        for compound in self .valid_compounds :
        # Load rows for the current compound
            compound_data =task_specific_data [task_specific_data ['compoundName']==compound ]

            if len (compound_data )>0 :
            # If Data, （orAccording toOne. Logical）
                row =compound_data .iloc [0 ]

                # Save SMILES
                self .compound_smiles [compound ]=row ['Drug']# Add: SaveSMILES

                # Build the property dictionary
                properties ={}
                for prop_col in self .property_columns :
                    value =row [prop_col ]
                    # ProcessingNaNValue
                    if pd .isna (value ):
                        properties [prop_col ]=None 
                    else :
                        properties [prop_col ]=value 

                self .compound_properties [compound ]=properties 
            else :
                print (f"Warning: no data found for compound {compound}")
                self .compound_smiles [compound ]=None # Add: No DataNone

        print (f"Tasks {self.target_task_names}: valid compound count = {len(self. valid_compounds)}")

    def __len__ (self ):
        return len (self .valid_compounds )

    def _load_image (self ,compoundName ):
        """Load a single image for one compound."""
        img_filename =compoundName +self .image_suffix 
        img_path =os .path .join (self .img_dir ,img_filename )

        try :
            image =Image .open (img_path ).convert ('RGB')
            if self .transform :
                image =self .transform (image )
            return image 
        except FileNotFoundError :
            print (f"Warning: image file is missing: {img_path}")
            return None 
        except Exception as e :
            print (f"Warning: failed to load image {img_path}: {e}")
            return None 
        finally :
        # Ensure image resources are released
            try :
                if 'image'in locals ()and hasattr (image ,'close'):
                    image .close ()
            except :
                pass 

    def __getitem__ (self ,idx ):
        if idx >=len (self .valid_compounds ):
            raise IndexError (f"Index {idx} out of range, valid compound count = {len(self.valid_compounds)}")

        compoundName =self .valid_compounds [idx ]
        image =self ._load_image (compoundName )
        if image is None :
            raise ValueError (f"ImageLoadFailed: {compoundName}")
        properties =self .compound_properties .get (compoundName ,{})
        smiles =self .compound_smiles .get (compoundName ,None )# Get cached SMILES

        # Return image, compound name, task id, properties, and SMILES
        return image ,compoundName ,self .task_id ,properties ,smiles 


def collate_fn_skip_invalid (batch ):
    """Collate a batch while preserving properties and SMILES."""
    # Images, compound names, task ids, property dicts, and SMILES
    images =[item [0 ]for item in batch ]
    compoundNames =[item [1 ]for item in batch ]
    task_names =[item [2 ]for item in batch ]
    properties_list =[item [3 ]for item in batch ]
    smiles_list =[item [4 ]for item in batch ]# Collect SMILES strings

    # Stack images into a batch tensor
    images_batch =torch .stack (images )

    # Reorganize property dictionaries by property name
    # Build one batched dict entry for each property column
    if properties_list :
    # Get
        all_property_keys =set ()
        for props in properties_list :
            if props is not None :# Make sure. None
                all_property_keys .update (props .keys ())

                # Create the batched property dictionary
        batch_properties ={}
        for key in all_property_keys :
            batch_properties [key ]=[]
            for props in properties_list :
                if props is not None and key in props :# InspectionandYes
                    batch_properties [key ].append (props .get (key ,None ))
                else :
                    batch_properties [key ].append (None )
    else :
        batch_properties ={}

    return images_batch ,compoundNames ,task_names ,batch_properties ,smiles_list 



