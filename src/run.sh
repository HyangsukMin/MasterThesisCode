#!/bin/bash

gpu_num=$1
dataset="mhealth"
beta=100

for epoch in $(seq 1 5)
do
    echo "##########################################################################################"
    echo "#" ${epoch} ":" ${dataset} ":RUNNING" $0 "with" $# "parameters and gpu_" ${gpu_num}
    echo "##########################################################################################"
    ./main.py --gpu_num ${gpu_num} --dataset ${dataset} --beta ${beta} --maxIters 2 --window 128 --shift 64 --savefig True --exp ${epoch} --epochs 100
done
