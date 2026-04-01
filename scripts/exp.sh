#!/bin/bash

# Configuration
DATASET="toyfashion_iq"
DATASET_PATH="/workspace/joel/fashionIQ"  # Update this to proper path
SPLIT="val"
BLIP_MODEL="Salesforce/blip2-flan-t5-xl"
CLIP_MODEL="ViT-B-32"
PROMPT="prompts.structural_modifier_prompt_fashion"

EXP_NAME="CIReVL_baseline"
LOG_FILE="logs/${EXP_NAME}.log"

echo "Running experiment -> ${LOG_FILE}"

python main.py \
    --exp-name ${EXP_NAME} \
    --dataset ${DATASET} \
    --dataset-path ${DATASET_PATH} \
    --split ${SPLIT} \
    --blip ${BLIP_MODEL} \
    --clip ${CLIP_MODEL} \
    --preprocess-type targetpad \
    --llm_prompt ${PROMPT} \
    > ${LOG_FILE} 2>&1


mkdir -p logs

echo "Experiments finished!"
