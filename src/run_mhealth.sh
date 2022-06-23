#!/bin/bash
gpu_num=$1
dataset=$2
commit=$3
ablation=$4

exp=1
dataset="mhealth"
beta=100

if [ $dataset -eq "mhealth" ]
then
    beta=100
    latent=10
    cluster_reassignment=50
elif [ $dataset -eq "pamap2" ]
then
    echo "DOWN"
else
    echo "KABOOM"
fi

window=128
shift=128
latent=10
cluster_reassignment=100
initializer="sequential"
kernel_size=3



#for epoch in $(seq 1 ${exp})
for epoch in 1
do
    echo "##########################################################################################"
    echo "#" ${epoch} ":" ${dataset} ":RUNNING" $0 "with" $# "parameters and gpu_" ${gpu_num}
    echo "##########################################################################################"
    ./main.py --gpu_num ${gpu_num} --dataset ${dataset} --beta ${beta} --window ${window} --shift ${shift} --exp ${epoch} --latent ${latent} --commit ${commit} --cluster_reassignment ${cluster_reassignment} --ablation ${ablation} --initializer ${initializer} --kernel_size ${kernel_size}
done
