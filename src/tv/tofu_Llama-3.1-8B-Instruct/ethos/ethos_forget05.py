import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import torch
from transformers import AutoModelForCausalLM
import warnings
import random

warnings.filterwarnings('ignore')
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num = 1
rank_num = '005'



load_path = "yourpath/src/tv/tofu_Llama-3.1-8B-Instruct/tofu_make_rank/pre_weight/forget05_llama_8b_it_ethos_"+rank_num+".pt"
V_residual = torch.load(load_path, map_location="cpu")  
print(f"Loaded V_forget from {load_path}")



rf_model_id = "yourpath/saves/finetune/tofu_Llama-3.1-8B-Instruct_full"
rf_model = AutoModelForCausalLM.from_pretrained(rf_model_id, use_flash_attention_2=True,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

to_erase_model = AutoModelForCausalLM.from_pretrained(rf_model_id, use_flash_attention_2=True,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)



with torch.no_grad():
    for (name, param_A), (_, param_B) in zip(to_erase_model.named_parameters(),
                                            rf_model.named_parameters()):
        param_A.copy_(param_B - V_residual[name].to(device) * num)

save_name = 'forget05_tv_alpha_'+str(num)

to_erase_model.save_pretrained('yourpath/src/tv/tofu_Llama-3.1-8B-Instruct/ethos/rank'+rank_num+'/save_name')
print("SAVE: ", num)


