#!/bin/bash

echo "dealing with group $1"

echo "mode: $2"

echo "itr number: $3"


for ((i=0; i<$3; i++))
do
    echo "getting stable state for iteration $i"
    CUDA_VISIBLE_DEVICES=2 python raisimGymTorch/env/envs/get_stable_state/get_stable_state.py -itr $i -group $1 -mode $2
done

# Loop 200 times
for ((i=0; i<$3; i++))
do
    echo "Running iteration $i"
    CUDA_VISIBLE_DEVICES=2 python raisimGymTorch/env/envs/ours_test/universal_eval.py -itr $i -group $1 -mode $2
done

