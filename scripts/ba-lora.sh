# run_training.sh

PRE_TRAINED_MODEL_PATH="/path/to/your/model"
OUTPUT_PATH="/path/to/your/output"
DATA_PATH="/path/to/your/data"

python ./train.py \
    --model_name_or_path $PRE_TRAINED_MODEL_PATH \
    --output_dir $OUTPUT_PATH \
    --adapter_name_or_path pissa_init \
    --init_lora_weights pissa \
    --data_path $DATA_PATH \
    --dataset_split "train[:100000]" \
    --dataset_field query response \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --lambda1=0.0001 \
    --lambda2=0.0003 \
    --lambda3=0.0001 \
    --report_to none
