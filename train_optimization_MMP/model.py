from __future__ import annotations 

import torch 
import torch .nn as nn 
import torch .nn .functional as F 


class FiLMLayer (nn .Module ):
    """Feature-wise Linear ModulationLayer"""
    def __init__ (self ,feature_dim :int ,condition_dim :int ):
        super ().__init__ ()
        self .feature_dim =feature_dim 
        self .film_generator =nn .Sequential (
        nn .Linear (condition_dim ,feature_dim *2 ),
        nn .ReLU (),
        nn .Linear (feature_dim *2 ,feature_dim *2 )
        )

    def forward (self ,features :torch .Tensor ,condition :torch .Tensor ):
        """ Args: features: (B, feature_dim) condition: (B, condition_dim) Returns: modulated features: (B, feature_dim) """
        film_params =self .film_generator (condition )
        gamma ,beta =film_params .chunk (2 ,dim =-1 )
        return features *(1 +gamma )+beta 


class TransformBlock (nn .Module ):
    """Convert Blocks；availableUse it. ConditionCLIFiLMMode，YesavailableMLPFragment"""
    def __init__ (
    self ,
    dim :int ,
    hidden_dim :int ,
    condition_dim :int =1 ,
    use_condition :bool =True 
    ):
        super ().__init__ ()
        self .use_condition =use_condition 
        self .fc1 =nn .Linear (dim ,hidden_dim )
        self .fc2 =nn .Linear (hidden_dim ,dim )
        self .norm =nn .LayerNorm (dim )
        if self .use_condition :
            self .film =FiLMLayer (dim ,condition_dim )
        else :
            self .film =None 
        self .dropout =nn .Dropout (0.1 )

    def forward (self ,x :torch .Tensor ,condition :torch .Tensor =None ):
        """ Args: x: (B, dim) condition: (B, condition_dim)；used in use_condition=False availableNone """
        residual =x 
        x =self .norm (x )
        x =F .relu (self .fc1 (x ))
        x =self .dropout (x )
        x =self .fc2 (x )
        if self .use_condition :
            if condition is None :
                raise ValueError ("Condition tensor must be provided when use_condition=True")
            x =self .film (x ,condition )
        return residual +x 




class PropertyExpert (nn .Module ):
    """SingleIt is. .. ：MLP + Multiple TransformBlock + OptionalAttention. + Layer。"""

    def __init__ (
    self ,
    rep_dim :int =512 ,
    hidden_dim :int =512 ,
    num_blocks :int =3 ,
    condition_dim :int =1 ,
    use_residual :bool =True ,
    use_attention :bool =True ,
    use_condition :bool =True ,
    ):
        super ().__init__ ()
        self .use_residual =use_residual 
        self .use_attention =use_attention 
        self .use_condition =use_condition 

        if self .use_condition :
            self .condition_encoders =nn .ModuleList ([
            nn .Sequential (
            nn .Linear (condition_dim ,hidden_dim //(2 **i )),
            nn .ReLU (),
            nn .Linear (hidden_dim //(2 **i ),hidden_dim )
            )
            for i in range (3 )
            ])

            self .condition_fusion =nn .Sequential (
            nn .Linear (hidden_dim *3 ,hidden_dim ),
            nn .ReLU (),
            nn .Linear (hidden_dim ,hidden_dim )
            )
        else :
            self .condition_encoders =None 
            self .condition_fusion =None 

        self .input_proj =nn .Linear (rep_dim ,rep_dim )
        self .transform_blocks =nn .ModuleList ([
        TransformBlock (
        rep_dim ,
        hidden_dim ,
        hidden_dim if self .use_condition else 1 ,
        use_condition =self .use_condition 
        )
        for _ in range (num_blocks )
        ])

        if self .use_attention :
            if self .use_condition :
                self .attention =nn .MultiheadAttention (
                embed_dim =rep_dim ,
                num_heads =8 ,
                dropout =0.1 ,
                kdim =hidden_dim ,
                vdim =hidden_dim ,
                )
            else :
                self .attention =nn .MultiheadAttention (
                embed_dim =rep_dim ,
                num_heads =8 ,
                dropout =0.1 
                )
            self .attention_norm =nn .LayerNorm (rep_dim )

        self .output_layers =nn .Sequential (
        nn .Linear (rep_dim ,rep_dim *2 ),
        nn .ReLU (),
        nn .Dropout (0.1 ),
        nn .Linear (rep_dim *2 ,rep_dim ),
        nn .LayerNorm (rep_dim )
        )

        self ._init_weights ()

    def _init_weights (self ):
        for m in self .modules ():
            if isinstance (m ,nn .Linear ):
                nn .init .xavier_uniform_ (m .weight )
                if m .bias is not None :
                    nn .init .zeros_ (m .bias )
            elif isinstance (m ,nn .LayerNorm ):
                nn .init .ones_ (m .weight )
                nn .init .zeros_ (m .bias )

    def _encode_condition (self ,ratio :torch .Tensor )->torch .Tensor :
        if ratio is None :
            raise ValueError ("Needs to be provided ratio CLICondition")
        if ratio .dim ()==1 :
            ratio =ratio .unsqueeze (-1 )
        feats =[encoder (ratio )for encoder in self .condition_encoders ]
        fused =torch .cat (feats ,dim =-1 )
        return self .condition_fusion (fused )

    def forward (self ,source_rep :torch .Tensor ,ratio :torch .Tensor =None )->torch .Tensor :
        condition =None 
        if self .use_condition :
            condition =self ._encode_condition (ratio )

        x =self .input_proj (source_rep )

        for block in self .transform_blocks :
            x =block (x ,condition )

        if self .use_attention :
            x_attn =x .unsqueeze (0 )
            if self .use_condition :
                condition_attn =condition .unsqueeze (0 )
                attn_out ,_ =self .attention (
                x_attn ,
                condition_attn ,
                condition_attn 
                )
            else :
                attn_out ,_ =self .attention (x_attn ,x_attn ,x_attn )
            attn_out =attn_out .squeeze (0 )
            x =self .attention_norm (x +attn_out )

        x =self .output_layers (x )

        if self .use_residual :
            return source_rep +x 
        return x 


class ConditionalRepAdjuster (nn .Module ):
    """ Essays（Local mask update）： - Every one. Experts（MLP+Transformer）； - Input 512 Vee. ，availableThe. .. Scope，OtherDimensionsValue； - Run/Destination PropertiesIt is. .. Experts；No additional linear projection，Promise. Dimensions。 """

    def __init__ (
    self ,
    rep_dim :int =512 ,
    hidden_dim :int =512 ,
    num_blocks :int =3 ,
    condition_dim :int =1 ,
    use_residual :bool =True ,
    use_attention :bool =True ,
    use_condition :bool =True ,
    property_names :list |None =None ,
    mask_span :int =20 ,
    mask_offset :int =0 ,
    ):
        super ().__init__ ()
        self .property_names =property_names or ["default"]
        self .property_to_idx ={name :idx for idx ,name in enumerate (self .property_names )}
        self .mask_span =int (mask_span )
        self .mask_offset =int (mask_offset )
        if self .mask_span <=0 :
            raise ValueError ("mask_span availableCount")
        if self .mask_offset <0 :
            raise ValueError ("mask_offset availableCount")
        total_span =self .mask_offset +self .mask_span *len (self .property_names )
        if total_span >rep_dim :
            raise ValueError (f"mask_offset + mask_span * num_properties More than rep_dim: {total_span} > {rep_dim}")

        self .experts =nn .ModuleDict ({
        name :PropertyExpert (
        rep_dim =rep_dim ,
        hidden_dim =hidden_dim ,
        num_blocks =num_blocks ,
        condition_dim =condition_dim ,
        use_residual =use_residual ,
        use_attention =use_attention ,
        use_condition =use_condition 
        )
        for name in self .property_names 
        })

        # Every one. Build
        masks =[]
        for idx in range (len (self .property_names )):
            mask =torch .zeros (rep_dim ,dtype =torch .float32 )
            start =self .mask_offset +idx *self .mask_span 
            end =start +self .mask_span 
            mask [start :end ]=1.0 
            masks .append (mask )
        self .register_buffer ("property_masks",torch .stack (masks ,dim =0 ))# (P, rep_dim)

    def forward (
    self ,
    source_rep :torch .Tensor ,
    ratio :torch .Tensor =None ,
    property_ids :torch .Tensor |None =None ,
    )->torch .Tensor :
        """ Args: source_rep: (B, rep_dim) ratio: (B,) or (B, 1) used inUse it. use_condition property_ids: (B,) long，Every one. Index；IfavailableNoneDefault0Experts """
        batch_size =source_rep .size (0 )
        device =source_rep .device 
        if property_ids is None :
            property_ids =torch .zeros (batch_size ,dtype =torch .long ,device =device )

        output =source_rep .clone ()
        for name ,expert in self .experts .items ():
            idx =self .property_to_idx [name ]
            mask_bool =self .property_masks [idx ].bool ()
            sample_mask =(property_ids ==idx )
            if not torch .any (sample_mask ):
                continue 
                # The RunExperts
            x_subset =source_rep [sample_mask ]
            ratio_subset =ratio [sample_mask ]if ratio is not None else None 
            expert_out =expert (x_subset ,ratio_subset )

            # Dimensions, OtherDimensionsValue
            current =output [sample_mask ]
            current [:,mask_bool ]=expert_out [:,mask_bool ]
            output [sample_mask ]=current 
        return output 
