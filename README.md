
# Entity-Level Knowledge Erasure via Orthogonal Basis and Adaptive Filtering

## Environment Setup
To replicate our experiments, please follow the steps below:

To set up the enviroment, run the following command:
```bash
# Environment setup
conda create -n unlearning python=3.11
conda activate unlearning
pip install .
pip install --no-build-isolation flash-attn==2.6.3

# Data setup
python setup_data.py  # saves/eval now contains evaluation results of the uploaded models
# Downloads log files with metric eval results (incl retain model logs) from the models 
# used in the supported benchmarks.
```

## Dataset
The `data` folder contains the `idk.jsonl` file, which is used when applying PO.
The TOFU dataset is automatically downloaded from Hugging Face \[[https://huggingface.co/datasets/locuslab/TOFU/viewer/full](https://huggingface.co/datasets/locuslab/TOFU/viewer/full)] when the command is executed.

## Training
### Full / Retain-only model: 
- You can specify `per_device_train_batch_size` and `gradient_accumulation_steps`.
For the Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct models, we separately train, save, and evaluate under three different settings: forget01, forget05, and forget10 (corresponding to retain99, retain95, and retain90).
```bash
bash 1_full_finetune_eval.sh
bash 1_retain_finetune_eval.sh
```

### 1. Gradient-based baselines:
Unlearning is applied to the previously trained full models using gradient-based baselines.

* The methods include GA, GD, KL, DPO\_KL, DPO\_GD, NPO\_KL, and NPO\_GD.
* You can configure parameters such as `alphas`, `betas`, `epoch`, `per_device_train_batch_size`, and `gradient_accumulation_steps`.

```bash
# Unlearn with gradient-based baselines
bash 2_unlearn_ga.sh
bash 2_unlearn_gd.sh
bash 2_unlearn_dpo_npo.sh

bash 2_unlearn_dpo_npo_eval.sh
bash 2_unlearn_gd_eval.sh
bash 2_unlearn_ga_eval.sh
```


### 2. TV
Inside `src/tv/model_name/original_tv`, you will find the files `original_tv_forget01.py`, `original_tv_forget05.py`, and `original_tv_forget10.py`.
Each script runs the task vector baseline method for the corresponding forget size.
The default setting is `num=1`, but it can be modified as desired.
Ensure that the value of alpha is updated accordingly for evaluation based on the chosen num.

Evaluation is performed using the `3_tofu_tv_eval.sh` script.

```bash
bash 3_ft_more.sh

# For Llama-3.1-8B-Instruct, run v_forget.py and v_retain.py in src/tv/tofu_Llama-3.1-8B-Instruct/tofu_make_rank/ in advance to generate the knowledge vectors.

# Run e.g. 'src/tv/model_name/original_tv/original_tv_forget01.py'

bash 3_tofu_tv_eval.sh
```


### 3. Ethos:
Inside `src/tv/model_name/ethos`, you will find the files `ethos_forget01.py`, `ethos_forget05.py`, and `ethos_forget10.py`.
Each script runs the task vector baseline method for the corresponding forget size.
The default setting is `num=1`, but it can be modified as desired.
Ensure that the value of alpha is updated accordingly for evaluation based on the chosen num.

The current threshold for ETHOS is set to 0.03 for Llama-3.2-1B-Instruct and 0.05 for Llama-3.1-8B-Instruct by default, but it can be adjusted according to user preference.
Evaluation is performed using the `3_tofu_tv_ethos_eval.sh` script.

```bash
bash 3_ft_more.sh

# For Llama-3.1-8B-Instruct, run v_forget.py and v_retain.py in src/tv/tofu_Llama-3.1-8B-Instruct/tofu_make_rank/ in advance to generate the knowledge vectors.

# Run e.g. ethos_rank_005_forget05.py located in the tofu_make_rank directory.

# Run e.g. 'src/tv/model_name/ethos/ethos_forget05.py'

bash 3_tofu_tv_ethos_eval.sh
```

### 4. ERASOR (Ours): 
Inside `src/tv/model_name/erasor`, you will find the files `erasor_forget01.py`, `erasor_forget05.py`, and `erasor_forget10.py`.
Each script runs the task vector baseline method for the corresponding forget size.
The default setting is `num=1`, but it can be modified as desired.
Ensure that the value of alpha is updated accordingly for evaluation based on the chosen num.

The EV for ERASOR is set to 0.9 by default, but it can be adjusted according to user preference.
Evaluation is performed using the `3_tofu_tv_erasor.sh` script.

```bash
bash 3_ft_more.sh

# For Llama-3.1-8B-Instruct, run v_forget.py and v_retain.py in src/tv/tofu_Llama-3.1-8B-Instruct/tofu_make_rank/ in advance to generate the knowledge vectors.

# Run e.g. erasor_ratio90_forget05.py located in the tofu_make_rank directory.

# Run e.g. 'src/tv/model_name/erasor/erasor_forget05.py'

bash 3_tofu_tv_erasor.sh
```

## Check result
The final results can be examined using the files in the `check_result` folder.
The current setup is based on the Forget01 setting.
To use Forget05 or Forget10, simply replace the corresponding file paths or configuration values with the desired setting.

* `tofu_llama-3.1-8b-instruct_all_es.ipynb` and `tofu_llama-3.2-1b-instruct_all_es.ipynb` present evaluation metrics for the TV, ETHOS, and ERASOR methods. These notebooks identify the best-performing models based on the highest ES-exact score on the forget set, under the constraint that the ES-exact score on the retain set of the full model remains at least 90%(or 95%).
* `tofu_llama-3.1-8b-instruct_gradient.ipynb` and `tofu_llama-3.2-1b-instruct_gradient.ipynb` perform a similar comparison for the gradient-based baselines. They search for the optimal models across different hyperparameter configurations and report the corresponding evaluation results.

---

### 🤝 Acknowledgements
- This repo is inspired from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). 
- The [TOFU](https://github.com/locuslab/tofu) benchmarks served as the foundation for our re-implementation. 
