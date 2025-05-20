import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_forget = "yourpath/src/tv/tofu_Llama-3.1-8B-Instruct/tofu_make_rank/pre_weight/forget05_llama_8b_it_forget_vector.pt"
path_retain = "yourpath/src/tv/tofu_Llama-3.1-8B-Instruct/tofu_make_rank/pre_weight/forget05_llama_8b_it_retain_vector.pt"

V_retain_dict = torch.load(path_retain)
print(f"Loaded V_retain from {path_retain}")
V_forget_dict = torch.load(path_forget)
print(f"Loaded V_forget from {path_forget}")


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
        rank_tmp = np.searchsorted(explained_variance_ratio, 1) + 1
        retain_basis = Vh[:rank_tmp]

        projection = forget_mat @ retain_basis.T @ retain_basis
        residual = forget_mat - projection

        if is_1d:
            residual = residual.squeeze(0)

    V_residual[name] = residual
    print(f"{name}")


final_save_path = "yourpath/src/tv/tofu_Llama-3.1-8B-Instruct/tofu_make_rank/pre_weight/forget05_llama_8b_it_erasor_ratiofull.pt"
torch.save(V_residual, final_save_path)
print(f"[Rank 0] Final merged result saved to {final_save_path}")
