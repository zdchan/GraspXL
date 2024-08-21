#!/bin/bash

echo "dealing with group $1"

echo "mode: $2"

echo "itr number: $3"


# Loop 200 times
for ((i=0; i<$3; i++))
do
    echo "Running iteration $i"
    CUDA_VISIBLE_DEVICES=7 python raisimGymTorch/env/envs/ours_test/universal_eval.py -itr $i -group $1 -mode $2
done

