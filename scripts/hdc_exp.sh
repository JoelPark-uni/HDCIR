#!/bin/bash

# Configuration
DATASET="toyfashion_iq"
DATASET_PATH="/workspace/joel/fashionIQ"  # Update this to proper path
SPLIT="val"
BLIP_MODEL="Salesforce/blip2-flan-t5-xl"
CLIP_MODEL="ViT-B-32"
PROMPT="prompts.structural_modifier_prompt_fashion"

HD_DIMS=(5000 10000 15000 20000 25000 30000)  # List of HD_DIM values to experiment with

mkdir -p logs

for DIM in "${HD_DIMS[@]}"; do
    EXP_NAME="hdc_exp_dim_${DIM}"
    LOG_FILE="logs/${EXP_NAME}.log"
    
    echo "Running experiment with HD_DIM=${DIM} -> ${LOG_FILE}"
    
    python main.py \
        --exp-name ${EXP_NAME} \
        --dataset ${DATASET} \
        --dataset-path ${DATASET_PATH} \
        --split ${SPLIT} \
        --blip ${BLIP_MODEL} \
        --clip ${CLIP_MODEL} \
        --use_hdc \
        --HD_DIM ${DIM} \
        --preprocess-type targetpad \
        --llm_prompt ${PROMPT} \
        > ${LOG_FILE} 2>&1
done

echo "Experiments finished!"
