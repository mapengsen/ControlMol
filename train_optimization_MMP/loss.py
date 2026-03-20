import torch 
import torch .nn as nn 
import torch .nn .functional as F 
from typing import Dict ,Optional ,Tuple 
from train_optimization_MMP .dataset import PROPERTY_CONFIG 


class OptimizationLoss (nn .Module ):
    """VersionLoss：Organisation L_target_property and L_invariant_property。"""

    def __init__ (
    self ,
    target_property_weight :float =1.0 ,
    invariant_property_weight :float =1.0 ,
    use_target_property_loss :bool =True ,
    use_invariant_property_loss :bool =True ,
    )->None :
        super ().__init__ ()
        self .target_property_weight =target_property_weight 
        self .invariant_property_weight =invariant_property_weight 
        self .use_target_property_loss =use_target_property_loss and target_property_weight !=0.0 
        self .use_invariant_property_loss =use_invariant_property_loss and invariant_property_weight !=0.0 

        print ("="*50 )
        print ("LossCount: ")
        print (f" {'✓' if self. use_target_property_loss else '✗'} Objective (L_target_property)")
        print (f" {'✓' if self. use_invariant_property_loss else '✗'} Objective (L_invariant_property)")
        print ("="*50 )

    def forward (
    self ,
    target_property_loss_per_sample :Optional [torch .Tensor ],
    invariant_loss_per_sample :Optional [torch .Tensor ],
    sample_weight :Optional [torch .Tensor ]=None ,
    )->Tuple [torch .Tensor ,Dict [str ,float ]]:

        if target_property_loss_per_sample is None and invariant_loss_per_sample is None :
            raise ValueError ("to target_property_loss_per_sample or invariant_loss_per_sample")

        device_ref =(
        target_property_loss_per_sample .device 
        if target_property_loss_per_sample is not None 
        else invariant_loss_per_sample .device 
        )
        batch_size =(
        target_property_loss_per_sample .size (0 )
        if target_property_loss_per_sample is not None 
        else invariant_loss_per_sample .size (0 )
        )

        if sample_weight is None :
            norm_weights =torch .ones (batch_size ,device =device_ref )
        else :
            norm_weights =sample_weight .to (device_ref )
            norm_weights =norm_weights /(norm_weights .mean ()+1e-8 )

        target_raw =torch .tensor (0.0 ,device =device_ref )
        target_weighted =torch .tensor (0.0 ,device =device_ref )
        if self .use_target_property_loss and target_property_loss_per_sample is not None :
            target_raw =target_property_loss_per_sample .mean ()
            target_weighted =torch .mean (target_property_loss_per_sample *norm_weights )

        invariant_raw =torch .tensor (0.0 ,device =device_ref )
        invariant_weighted =torch .tensor (0.0 ,device =device_ref )
        if self .use_invariant_property_loss and invariant_loss_per_sample is not None :
            invariant_raw =invariant_loss_per_sample .mean ()
            invariant_weighted =torch .mean (invariant_loss_per_sample *norm_weights )

        total_loss =torch .tensor (0.0 ,device =device_ref )
        total_raw =torch .tensor (0.0 ,device =device_ref )

        if self .use_target_property_loss and target_property_loss_per_sample is not None :
            total_loss =total_loss +self .target_property_weight *target_weighted 
            total_raw =total_raw +target_raw 
        if self .use_invariant_property_loss and invariant_loss_per_sample is not None :
            total_loss =total_loss +self .invariant_property_weight *invariant_weighted 
            total_raw =total_raw +invariant_raw 

        loss_dict ={
        "total_weighted":total_loss .item (),
        "total_raw":total_raw .item (),
        "target_property_raw":(
        target_raw .item ()if self .use_target_property_loss and target_property_loss_per_sample is not None else 0.0 
        ),
        "target_property_weighted":(
        self .target_property_weight *target_weighted 
        ).item ()if self .use_target_property_loss and target_property_loss_per_sample is not None else 0.0 ,
        "invariant_raw":(
        invariant_raw .item ()if self .use_invariant_property_loss and invariant_loss_per_sample is not None else 0.0 
        ),
        "invariant_weighted":(
        self .invariant_property_weight *invariant_weighted 
        ).item ()if self .use_invariant_property_loss and invariant_loss_per_sample is not None else 0.0 ,
        }

        return total_loss ,loss_dict 


def train_step (
model :nn .Module ,
batch :Dict [str ,torch .Tensor ],
criterion :OptimizationLoss ,
device :torch .device ,
property_predictor :Optional [nn .Module ]=None ,
property_task_configs :Optional [Dict [str ,Dict [str ,str ]]]=None ,
property_normalizers :Optional [Dict [str ,Optional [object ]]]=None ,
target_property_name :Optional [str ]=None ,
preserve_properties :Optional [list ]=None ,
property_names_map :Optional [list ]=None ,
)->Tuple [torch .Tensor ,Dict [str ,float ]]:

    source_rep =batch ["source_rep"].to (device )
    ratio =batch .get ("ratio")
    if ratio is not None :
        ratio =ratio .to (device )
    property_ids =batch .get ("property_id")
    if property_ids is not None :
        property_ids =property_ids .to (device )
    sample_weight =batch .get ("quality_weight")
    if sample_weight is not None :
        sample_weight =sample_weight .to (device )

    target_property_start =batch .get ("target_property_start")
    preserve_values =batch .get ("preserve_properties",{})
    if target_property_start is not None :
        target_property_start =target_property_start .to (device )
    if preserve_values :
        preserve_values ={prop :tensor .to (device )for prop ,tensor in preserve_values .items ()}
    else :
        preserve_values ={}

    predicted_rep =model (source_rep ,ratio ,property_ids =property_ids )

    target_property_loss_per_sample =None 
    invariant_loss_per_sample =None 

    if (
    property_predictor is not None 
    and property_task_configs is not None 
    and property_normalizers is not None 
    ):
        property_outputs =property_predictor (predicted_rep )
        if property_ids is None and target_property_name is not None :
            property_names_map =[target_property_name ]
            property_ids =torch .zeros (source_rep .size (0 ),dtype =torch .long ,device =device )

            # ObjectiveLoss: ClassesUse it. preferred_transition. final, ReturnsUse it. start + preferred_delta
        target_property_loss_per_sample =torch .zeros (source_rep .size (0 ),device =device )
        if property_ids is not None and property_names_map is not None :
            for idx ,prop_name in enumerate (property_names_map ):
                mask =(property_ids ==idx )
                if not torch .any (mask ):
                    continue 
                logits =property_outputs .get (prop_name )
                if logits is None :
                    continue 
                cfg =property_task_configs .get (prop_name ,{})
                prop_cfg =PROPERTY_CONFIG .get (prop_name ,{})
                prop_type =cfg .get ("type",prop_cfg .get ("type","regression"))

                if prop_type =="classification":
                    transition =prop_cfg .get ("preferred_transition",{})
                    final_target =float (transition .get ("final",1.0 ))
                    labels =torch .full_like (logits [mask ],final_target ,dtype =torch .float32 ).clamp (0.0 ,1.0 )
                    loss_val =F .binary_cross_entropy_with_logits (logits [mask ],labels ,reduction ="none")
                    target_property_loss_per_sample [mask ]=loss_val .squeeze ()
                else :
                    normalizer =property_normalizers .get (prop_name )
                    denorm_pred =normalizer .denormalize (logits [mask ])if normalizer is not None else logits [mask ]
                    start_vals =target_property_start [mask ].float ()if target_property_start is not None else torch .zeros_like (denorm_pred )
                    required_delta =float (prop_cfg .get ("preferred_delta",cfg .get ("preferred_delta",0.0 )))
                    desired =start_vals +required_delta 
                    loss_val =F .mse_loss (denorm_pred .squeeze (),desired .squeeze (),reduction ="none")
                    target_property_loss_per_sample [mask ]=loss_val 

        invariant_losses =[]
        if preserve_properties is None :
            preserve_properties =list (preserve_values .keys ())
        for prop in preserve_properties or []:
            if prop not in property_task_configs :
                continue 
            logits =property_outputs .get (prop )
            if logits is None or prop not in preserve_values :
                continue 
            cfg =property_task_configs [prop ]
            targets =preserve_values [prop ].float ()
            if cfg .get ("type")=="classification":
                targets =targets .clamp (0.0 ,1.0 )
                loss_val =F .binary_cross_entropy_with_logits (logits ,targets ,reduction ="none")
            else :
                normalizer =property_normalizers .get (prop )
                denorm_pred =normalizer .denormalize (logits )if normalizer is not None else logits 
                loss_val =F .mse_loss (denorm_pred ,targets ,reduction ="none")
            invariant_losses .append (loss_val )

        if invariant_losses :
            invariant_loss_per_sample =torch .stack (invariant_losses ,dim =0 ).mean (dim =0 )

    loss ,loss_dict =criterion (
    target_property_loss_per_sample =target_property_loss_per_sample ,
    invariant_loss_per_sample =invariant_loss_per_sample ,
    sample_weight =sample_weight ,
    )

    return loss ,loss_dict 
