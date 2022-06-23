#!/bin/bash

gpu_num=$1
dataset=$2
commit=$3
ablation=$4


exp=5
window=2048
shift=2048
initializer="sequential"
kernel_size=3
maxIters=2
pretrain_epoch=1

if [ -z "$ablation" ];then
   ablation="None"
fi

if [ $ablation == "noPseudo" ]
then
    epochs=10
    maxIters=10
elif [ $ablation == "noICC" ]
then
    epochs=100
else
    epochs=100
fi


if [ $dataset == "mhealth" ]; then
    echo "DATASET IS MHEALTH. & ABLATION IS ${ablation}"
    window=2048
    shift=2048
    beta=300
    n_latent=10
    cluster_reassignment=100
    n_dilation=8
    dilation_exp=2
    sampling_factor=8
elif [ $dataset == "pamap2" ]; then
    echo "DATASET IS PAMAP2. & ABLATION IS ${ablation}"
    window=2048
    shift=2048
<<<<<<< HEAD
    beta=200
=======
    beta=100
>>>>>>> 2f5dc7987a42190942903bb0ca173411330d4670
    n_latent=20
    cluster_reassignment=100
    n_dilation=8
    dilation_exp=2
    sampling_factor=16
elif [ $dataset == "skoda" ]; then
    echo "DATASET IS SKODA. & ABLATION IS ${ablation}"
    window=2048
    shift=2048
    beta=200
    n_latent=20
    cluster_reassignment=100
    n_dilation=8
    dilation_exp=4
    sampling_factor=8
else
    echo "Dataset Not Found."
    exit
fi


for epoch in $(seq 1 ${exp})
#for epoch in 1
do
    echo "##########################################################################################"
    echo "#" ${epoch} ":" ${dataset} ":RUNNING" $0 "with" $# "parameters and gpu_" ${gpu_num}
    echo "##########################################################################################"
    if [ $exp -gt 1 ];
    then
	commits="${commit}_${epoch}"
	./main.py --gpu_num ${gpu_num} --dataset ${dataset} --beta ${beta} --window ${window} --shift ${shift} --exp ${epoch} --n_latent ${n_latent} --commit ${commits} --cluster_reassignment ${cluster_reassignment} --ablation ${ablation} --initializer ${initializer} --kernel_size ${kernel_size} --dilation_exp ${dilation_exp} --epochs ${epochs} --maxIters ${maxIters} --pretrain_epoch ${pretrain_epoch}
    else
	./main.py --gpu_num ${gpu_num} --dataset ${dataset} --beta ${beta} --window ${window} --shift ${shift} --exp ${epoch} --n_latent ${n_latent} --commit ${commit} --cluster_reassignment ${cluster_reassignment} --ablation ${ablation} --initializer ${initializer} --kernel_size ${kernel_size}  --dilation_exp ${dilation_exp} --epochs ${epochs} --maxIters ${maxIters} --pretrain_epoch ${pretrain_epoch}
    fi
done
