#!/bin/bash


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.1-8B-Instruct"
)
trainers_experiments=(
    "NPO unlearn/tofu/default.yaml"
    "DPO unlearn/tofu/idk.yaml"
)
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

per_device_train_batch_size=4 
gradient_accumulation_steps=4

ckpts=(
    "1"
    "2"
    "3"
    "4"
)

alphas=(
    "1"
    "2"
    "10"
    "20"
)

betas=(
    "0.1"
    "1"
    "10"
)

epoch=4

retain_loss_types=("NLL" "KL")

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            trainer=$(echo $trainer_experiment | cut -d' ' -f1)
            experiment=$(echo $trainer_experiment | cut -d' ' -f2)

            for retain_loss_type in "${retain_loss_types[@]}"; do
                if [ "$retain_loss_type" == "NLL" ]; then
                    suffix="GDR"
                else
                    suffix="KL"
                fi

                for ckpt in "${ckpts[@]}"; do
                    for alpha in "${alphas[@]}"; do
                        for beta in "${betas[@]}"; do

                            task_name=tofu_${model}/${forget_split}/${trainer}_${suffix}/tofu_${model}_${forget_split}_${trainer}_epoch${epoch}_alpha${alpha}_beta${beta}_${suffix}/checkpoint-${ckpt}
                            model_path=open-unlearning/tofu_${model}_full
                            echo "${task_name}: Evaluating ${model_path} using ${trainer} (${retain_loss_type})"

                            # Eval
                            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                            experiment=eval/tofu/default.yaml \
                            forget_split=${forget_split} \
                            holdout_split=${holdout_split} \
                            model=${model} \
                            task_name=${task_name} \
                            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                            paths.output_dir=saves/unlearn/${task_name}/evals \
                            retain_logs_path=saves/finetune/tofu_${model}_${retain_split}/evals/TOFU_EVAL.json

                            # Delete safetensors
                            target_dir=saves/unlearn/${task_name}
                            echo "[INFO] Deleting .safetensors in: $target_dir"
                            find "$target_dir" -type f -name "*.safetensors" -exec rm -f {} +
                        done
                    done
                done
            done
        done
    done
done
