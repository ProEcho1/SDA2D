#!/bin/bash

echo "Training Start!"

datasets_name=('SMD' 'MSL' 'PSM' 'SWaT' 'SMAP')

for dataset in "${datasets_name[@]}"; do
  echo "Dataset: $dataset"
  python main.py --dataset "$dataset"
  wait
done

echo "Training Finished."
