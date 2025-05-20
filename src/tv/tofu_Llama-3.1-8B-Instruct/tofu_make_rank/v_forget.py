import torch
from transformers import AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rff_model_id = "yourpath/saves/unlearn/tofu_Llama-3.1-8B-Instruct/tofu_Llama-3.1-8B-Instruct_forget01_GradDescent_forget" # Can change to forget05, forget10
rff_model = AutoModelForCausalLM.from_pretrained(rff_model_id, use_flash_attention_2=True,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
                                         
rf_model_id = "yourpath/saves/finetune/tofu_Llama-3.1-8B-Instruct_full"
rf_model = AutoModelForCausalLM.from_pretrained(rf_model_id, use_flash_attention_2=True,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

V_forget_dict = {}

with torch.no_grad():
    for (name, param_A), (_, param_B) in zip(rf_model.named_parameters(),
                                             rff_model.named_parameters()):
        V_forget_dict[name] = (param_B.cpu() - param_A.cpu())
        print(name)


final_save_path = "yourpath/src/tv/tofu_Llama-3.1-8B-Instruct/tofu_make_rank/pre_weight/forget01_llama_8b_it_forget_vector.pt" # Can change to forget05, forget10
torch.save(V_forget_dict, final_save_path)
print(f"[Rank 0] Final merged result saved to {final_save_path}")