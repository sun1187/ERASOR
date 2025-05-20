import torch
from transformers import AutoModelForCausalLM

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_forget = "yourpath/src/tv/tofu_Llama-3.1-8B-Instruct/tofu_make_rank/pre_weight/forget10_llama_8b_it_forget_vector.pt"
rf_model_id = "yourpath/saves/finetune/tofu_Llama-3.1-8B-Instruct_full"
rf_model = AutoModelForCausalLM.from_pretrained(rf_model_id, use_flash_attention_2=True,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

V_forget_dict = torch.load(path_forget)
print(f"Loaded V_forget from {path_forget}")

RF_dict = {}
with torch.no_grad():
    for name, param_A in rf_model.named_parameters():
        RF_dict[name] = param_A.cpu()

V_ethos = {}
for name, param in V_forget_dict.items():
    forget_mat = V_forget_dict[name].float()
    retain_mat = RF_dict[name].float()

    is_1d = False
    if forget_mat.dim() == 1:
        forget_mat = forget_mat.unsqueeze(0)
        retain_mat = retain_mat.unsqueeze(0)
        is_1d = True

    U, S, Vh = torch.linalg.svd(retain_mat, full_matrices=False)

    S_prime = U.T @ forget_mat @ Vh.T

    thres = S_prime.max() * 0.05
    S_prime = torch.where(
        (S_prime < thres) & (S_prime > -thres),
        torch.zeros_like(S_prime),
        S_prime,
    )

    new_wd = U @ S_prime @ Vh
    if is_1d:
        new_wd = new_wd.squeeze(0)

    V_ethos[name] = new_wd
    print(name)


final_save_path = "yourpath/src/tv/tofu_Llama-3.1-8B-Instruct/tofu_make_rank/pre_weight/forget10_llama_8b_it_ethos_005.pt"
torch.save(V_ethos, final_save_path)
print(f"[Rank 0] Final merged result saved to {final_save_path}")
