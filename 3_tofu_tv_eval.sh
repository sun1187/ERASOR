#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.1-8B-Instruct"
)

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

alphas=(
    "0"
    "1"
    "2"
    "3"
)

tasks=(
    original_tv
)


for model in "${models[@]}"; do

    # Evaluate the full models on each forget split
    for split in "${splits[@]}"; do

        for task in "${tasks[@]}"; do

            for alpha in "${alphas[@]}"; do
                forget_split=$(echo $split | cut -d' ' -f1)
                holdout_split=$(echo $split | cut -d' ' -f2)
                retain_split=$(echo $split | cut -d' ' -f3)

                CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
                forget_split=${forget_split} \
                holdout_split=${holdout_split} \
                task_name=tofu_${model}_${forget_split} \
                model=${model} \
                model.model_args.pretrained_model_name_or_path=src/tv/tofu_${model}/${task}/weight/${forget_split}_alpha_${alpha} \
                retain_logs_path=saves/finetune/tofu_${model}_${retain_split}/evals/TOFU_EVAL.json \
                paths.output_dir=src/tv/tofu_${model}/${task}/weight/${forget_split}_alpha_${alpha}/evals

                target_dir=src/tv/tofu_${model}/${task}/weight/${forget_split}_alpha_${alpha}
                echo "[INFO] Deleting .safetensors in: $target_dir"
                find "$target_dir" -type f -name "*.safetensors" -exec rm -f {} +
            done
        done
    done
done
