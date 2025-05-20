import numpy as np
import torch
from transformers import AutoModelForCausalLM

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


rff_model_id = "yourpath/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_GradDescent_forget"
rff_model = AutoModelForCausalLM.from_pretrained(rff_model_id, use_flash_attention_2=True,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

rrf_model_id = "yourpath/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_GradDescent_retain"
rrf_model = AutoModelForCausalLM.from_pretrained(rrf_model_id, use_flash_attention_2=True,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
                                                 
rf_model_id = "yourpath/saves/finetune/tofu_Llama-3.2-1B-Instruct_full"
rf_model = AutoModelForCausalLM.from_pretrained(rf_model_id, use_flash_attention_2=True,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

to_erase_model = AutoModelForCausalLM.from_pretrained(rf_model_id, use_flash_attention_2=True,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)


V_forget_dict = {}
with torch.no_grad():
    for (name, param_A), (_, param_B) in zip(rf_model.named_parameters(),
                                             rff_model.named_parameters()):
        V_forget_dict[name] = param_B - param_A  

V_retain_dict = {}
with torch.no_grad():
    for (name, param_A), (_, param_B) in zip(rf_model.named_parameters(),
                                             rrf_model.named_parameters()):
        V_retain_dict[name] = param_B - param_A  


V_residual = {}

for name in V_forget_dict.keys():
    
    forget_mat = V_forget_dict[name].float().cpu()
    retain_mat = V_retain_dict[name].float().cpu()

    is_1d = False
    if retain_mat.dim() == 1:
        retain_mat = retain_mat.unsqueeze(0)
        is_1d = True

    U, S, Vh = torch.linalg.svd(retain_mat, full_matrices=False)
    S_np = S.numpy()

    S_squared = S_np**2
    total_variance = np.sum(S_squared)
    if total_variance == 0:
        
        print(f"[SKIP] {name} (S.sum()==0)")
        residual = forget_mat
        
    else:
        explained_variance_ratio = np.cumsum(S_np**2) / np.sum(S_np**2)
        rank_tmp = np.searchsorted(explained_variance_ratio, 0.90) + 1
        retain_basis = Vh[:rank_tmp]

        projection = forget_mat @ retain_basis.T @ retain_basis
        residual = forget_mat - projection

        if is_1d:
            residual = residual.squeeze(0)

    V_residual[name] = residual
    print(f"{name}")

final_save_path = "yourpath/src/tv/tofu_Llama-3.2-1B-Instruct/tofu_make_rank/pre_weight/forget10_llama3_1b_erasor_ratio90.pt"
torch.save(V_residual, final_save_path)
print(f"[Rank 0] Final merged result saved to {final_save_path}")
