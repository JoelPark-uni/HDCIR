#!/bin/bash

# Configuration
# DATASET="toyfashion_iq"
DATASET="circo"
DATASET_PATH="/workspace/joel/CIRCO"  # Update this to proper path
SPLIT="val"
BLIP_MODEL="Salesforce/blip2-flan-t5-xl"
CLIP_MODEL="ViT-B-32"
# PROMPT="prompts.structural_modifier_prompt_fashion"
PROMPT="prompts.structural_modifier_prompt"

echo "Running experiment "
EXP_NAME="circo_splice_v2"
LOG_FILE="logs/${EXP_NAME}_0513.log"

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
    --use_splice \
    --active-intents negation addition direct_addressing compare_change spatial_relations_background viewpoint comparative_statement cardinality \
    --cache-path "logs/${EXP_NAME}_cache.pt" \
    > ${LOG_FILE} 2>&1


    # --active-intents negation spatial_relations_background viewpoint \
    # --active-intents comparative_statement cardinality \
    # --active-intents negation addition direct_addressing compare_change spatial_relations_background viewpoint comparative_statement cardinality \


# threshold_list=(100.0 200.0 300.0)

# for threshold in "${threshold_list[@]}"; do
#     echo "Running experiment with HDClassifier threshold: ${threshold}"
#     EXP_NAME="toy_circo_hdclassifier${threshold}_exp"
#     LOG_FILE="logs/${EXP_NAME}.log"

#     echo "Running experiment -> ${LOG_FILE}"

#     python main.py \
#         --exp-name ${EXP_NAME} \
#         --dataset ${DATASET} \
#         --dataset-path ${DATASET_PATH} \
#         --split ${SPLIT} \
#         --blip ${BLIP_MODEL} \
#         --clip ${CLIP_MODEL} \
#         --preprocess-type targetpad \
#         --llm_prompt ${PROMPT} \
#         --use_hdclassifier \
#         --hdclassifier_threshold ${threshold} \
#         > ${LOG_FILE} 2>&1
# done

echo "Experiments finished!"
