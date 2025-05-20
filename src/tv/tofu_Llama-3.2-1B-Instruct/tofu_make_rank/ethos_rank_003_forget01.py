import torch
import torch
from transformers import AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


rff_model_id = "your_path/saves/unlearn/tofu_Llama-3.2-1B-Instruct/tofu_Llama-3.2-1B-Instruct_forget01_GradDescent_forget"
rff_model = AutoModelForCausalLM.from_pretrained(rff_model_id, use_flash_attention_2=True,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

rf_model_id = "your_path/saves/finetune/tofu_Llama-3.2-1B-Instruct_full"
rf_model = AutoModelForCausalLM.from_pretrained(rf_model_id, use_flash_attention_2=True,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

to_erase_model = AutoModelForCausalLM.from_pretrained(rf_model_id, use_flash_attention_2=True,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)


V_forget_dict = {}
RF_dict = {}
with torch.no_grad():
    for (name, param_A), (_, param_B) in zip(rf_model.named_parameters(),
                                             rff_model.named_parameters()):
        V_forget_dict[name] = param_B - param_A  
        RF_dict[name] = param_A

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

    thres = S_prime.max() * 0.03
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


final_save_path = "your_path/src/tv/tofu_Llama-3.2-1B-Instruct/tofu_make_rank/pre_weight/forget01_llama3_1b_ethos_003.pt"
torch.save(V_ethos, final_save_path)
print(f"Final merged result saved to {final_save_path}")
