#!/bin/bash

gpu_num=$1
ablation="None"
commit=$2
exp=1

dataset="skoda"
beta=300
ablation="None"
window=128
shift=128
latent=10
cluster_reassignment=200
initializer="sequential"
kernel_size=27

#for epoch in $(seq 1 ${exp})
for epoch in 1
do
    echo "##########################################################################################"
    echo "#" ${epoch} ":" ${dataset} ":RUNNING" $0 "with" $# "parameters and gpu_" ${gpu_num}
    echo "##########################################################################################"
    ./main.py --gpu_num ${gpu_num} --dataset ${dataset} --beta ${beta} --window ${window} --shift ${shift} --exp ${epoch} --latent ${latent} --commit ${commit} --cluster_reassignment ${cluster_reassignment} --ablation ${ablation} --initializer ${initializer} --kernel_size ${kernel_size}
    
done
