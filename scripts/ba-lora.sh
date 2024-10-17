RESIDUAL_MODEL="/path/to/your/model"
OUTPUT="/path/to/your/output"
DATA_PATH="/path/to/your/data"

python src/train.py \
    --model_name_or_path $RESIDUAL_MODEL \
    --output_dir $OUTPUT \
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
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --lcr=0.0001 \
    --ldr=0.0003 \
    --lsvd-loss-weight=0.0001 \
    --report_to none
