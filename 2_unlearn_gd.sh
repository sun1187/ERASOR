#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.1-8B-Instruct"
)
trainers_experiments=(
    "GradDiff unlearn/tofu/default.yaml"
)
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

per_device_train_batch_size=4
gradient_accumulation_steps=4

epochs=(
    "4"
)

alphas=(
    "1"
    "2"
    "10"
    "20"
)

retain_loss_types=(
    "NLL"
    "KL"
)

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

                for epoch in "${epochs[@]}"; do
                    for alpha in "${alphas[@]}"; do
                        task_name=tofu_${model}/${forget_split}/${trainer}_${suffix}/tofu_${model}_${forget_split}_${trainer}_epoch${epoch}_alpha${alpha}_${suffix}
                        model_path=open-unlearning/tofu_${model}_full
                        echo "${task_name}: Unlearning ${model_path} using ${trainer} (${retain_loss_type})"

                        # Unlearn
                        CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
                        src/train.py --config-name=unlearn.yaml \
                        experiment=${experiment} \
                        trainer=${trainer} \
                        task_name=${task_name} \
                        model=${model} \
                        forget_split=${forget_split} \
                        retain_split=${retain_split} \
                        model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_full \
                        retain_logs_path=saves/finetune/tofu_${model}_${retain_split}/evals/TOFU_EVAL.json \
                        trainer.method_args.alpha=${alpha} \
                        trainer.method_args.retain_loss_type=${retain_loss_type} \
                        trainer.args.num_train_epochs=${epoch} \
                        trainer.args.save_strategy=epoch \
                        trainer.args.logging_strategy=epoch \
                        trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                        trainer.args.per_device_eval_batch_size=$per_device_train_batch_size \
                        trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                        trainer.args.ddp_find_unused_parameters=true \
                        trainer.args.gradient_checkpointing=true
                    done
                done
            done
        done
    done
done
