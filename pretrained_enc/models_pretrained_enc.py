import torch
import torchvision


## ================================== CGIP ==================================
def load_pretrained_CGIP_image_ckpt(model, ckpt_path):
    """
    Load Pretrain ResNet18 for Encoder
    @param model: The instantiated model
    @param ckpt_path:
    @return:
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # Load the checkpoint parameter to the ResNet model
    ckp_keys = list(checkpoint['model_state_dict1'])
    cur_keys = list(model.state_dict())
    model_sd = model.state_dict()
    ckp_keys = ckp_keys[:120]

    for ckp_key, cur_key in zip(ckp_keys, cur_keys):
        model_sd[cur_key] = checkpoint['model_state_dict1'][ckp_key]

    model.load_state_dict(model_sd)
    print("======================== Load Image SSL Encoder from CGIP... ========================")

    return model



def CGIP_image_model():
    model = torchvision.models.resnet18(pretrained=False)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.name = 'CGIP'

    return model
