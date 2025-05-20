#!/bin/bash


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.1-8B-Instruct"
)
trainers_experiments=(
    "GradAscent unlearn/tofu/default.yaml"
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
for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            for epoch in "${epochs[@]}"; do
                trainer=$(echo $trainer_experiment | cut -d' ' -f1)
                experiment=$(echo $trainer_experiment | cut -d' ' -f2)
                
                task_name=tofu_${model}/${forget_split}/${trainer}/tofu_${model}_${forget_split}_${trainer}_epoch${epoch}
                model_path=open-unlearning/tofu_${model}_full
                echo ${task_name}: Unlearning ${model_path} using ${trainer}

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
                trainer.args.save_strategy=epoch \
                trainer.args.logging_strategy=epoch \
                trainer.args.num_train_epochs=${epoch} \
                trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                trainer.args.per_device_eval_batch_size=$per_device_train_batch_size \
                trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true
            done
        done
    done
done
