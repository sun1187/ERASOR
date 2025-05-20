#!/bin/bash


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"



# Train the full and retain-only models
bash 1_full_finetune_eval.sh
bash 1_retain_finetune_eval.sh


# Unlearn with gradient-based baselines
bash 2_unlearn_ga.sh
bash 2_unlearn_gd.sh
bash 2_unlearn_dpo_npo.sh

bash 2_unlearn_dpo_npo_eval.sh
bash 2_unlearn_gd_eval.sh
bash 2_unlearn_ga_eval.sh

bash 3_ft_more.sh
