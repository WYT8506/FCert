import os

Ks= [15]
Cs= [15]
DATASET_TYPEs =["cubirds200", "cifarfs", "tiered_imagenet"]
MODEL_TYPEs = ["CLIP"]
CERTIFICATION_TYPE = "group"

for MODEL_TYPE in MODEL_TYPEs:
    for i in range(len(Ks)):
        K = Ks[i]
        C = Cs[i]
        for j in range(len(DATASET_TYPEs)):

            DATASET_TYPE = DATASET_TYPEs[j]

            cmd = f'python -u main.py \
            --dataset_type {DATASET_TYPE} \
            --model_type {MODEL_TYPE} \
            --certification_type {CERTIFICATION_TYPE} \
            --file_path ./output/{CERTIFICATION_TYPE}_K={K}_C={C}_{DATASET_TYPE}_{MODEL_TYPE}.json \
            --classes_per_it_val {C} \
            --num_query_val {1} \
            --num_support_val {K} '

            os.system(cmd)

