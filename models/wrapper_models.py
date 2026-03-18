# import random
# import torch
# from torch import nn

# class WrapperModel_SD3_ControlNet_with_Adapter(nn.Module):
#     def __init__(self, controlnet, adapter, **kwargs):
#         super(WrapperModel_SD3_ControlNet_with_Adapter, self).__init__()
#         self.controlnet = controlnet
#         self.adapter = adapter

#     def forward(self, noisy_model_input, timestep, prompt_embeds, controlnet_pooled_projections, controlnet_cond, text_embeds, **kwargs):
#         # text embed shape: [b, 128, 1472]
#         text_features = self.adapter(text_embeds) # [b, 128, 4096]
#         # controlnet
#         control_block_samples = self.controlnet(
#             hidden_states=noisy_model_input,
#             timestep=timestep,
#             encoder_hidden_states=text_features,
#             pooled_projections=controlnet_pooled_projections,
#             controlnet_cond=controlnet_cond,
#             return_dict=False,
#         )[0]
#         return control_block_samples

import random
import torch
from torch import nn

class WrapperModel_SD3_ControlNet_with_Adapter(nn.Module):
    def __init__(self, controlnet, adapter, **kwargs):
        super(WrapperModel_SD3_ControlNet_with_Adapter, self).__init__()
        self.controlnet = controlnet
        self.adapter = adapter

    def forward(self, noisy_model_input, timestep, prompt_embeds, controlnet_pooled_projections, controlnet_cond, text_embeds, incoming_features=None, **kwargs):
        """
        Modified forward to support Interactive Mode (CalliEdit-V2)
        Args:
            incoming_features: List of tensors from the Background Stream (SceneGenNet)
        """
        # 1. Adapt text embeddings
        # text embed shape: [b, 128, 1472] -> [b, 128, 4096]
        text_features = self.adapter(text_embeds) 
        
        # 2. Run ControlNet in Receiver Mode
        # We use the new 'forward_as_receiver' method to inject incoming_features into TSIM
        control_block_samples = self.controlnet.forward_as_receiver(
            hidden_states=noisy_model_input,
            timestep=timestep,
            encoder_hidden_states=text_features,
            pooled_projections=controlnet_pooled_projections,
            controlnet_cond=controlnet_cond,
            incoming_features=incoming_features, # <--- Pass the background features here
            return_dict=False,
        )[0]
        
        return control_block_samples