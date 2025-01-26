#!/bin/bash

# python3 sem_test.py \
#   --target_directory_path ../generative-models/group_3/images \
#   --top_n_json sim_group_1_3.json \
#   --input_csv_path shap_group1_10000_s.csv \
#   --output_csv_path shap_group3_10000_s.csv
for i in 3 4 5 6 7 8 9
  do
  for n in 10000 10500 11000 11500 12000
  do
    python3 knn_cl.py \
      --target_directory_path ../generative-models/group_${i}/images \
      --top_n_json ../shapcal/sim_group_1_${i}.json \
      --input_csv_path copyrightloss_1_${n}.csv \
      --output_csv_path copyrightloss_${i}_${n}.csv
  done
done