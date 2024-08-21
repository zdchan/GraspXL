#!/bin/bash


echo "dealing with group $1, mode $2"
for ((i=0; i<40; i++))
do
    echo "Running iteration $i"
    CUDA_VISIBLE_DEVICES=7 python raisimGymTorch/env/envs/get_stable_state/get_stable_state.py -itr $i -group $1 -mode $2
done