#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.1-8B-Instruct"
)
per_device_train_batch_size=4 
gradient_accumulation_steps=4 

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)


for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    
    for model in "${models[@]}"; do
        CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
        src/train.py experiment=finetune/tofu/default.yaml \
        task_name=tofu_${model}_${retain_split} \
        model=${model} \
        data/datasets@data.train=TOFU_QA_retain \
        data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true

    
        CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=tofu_${model}_${retain_split} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_${retain_split} \
        paths.output_dir=saves/finetune/tofu_${model}_${retain_split}/evals
    done
done
