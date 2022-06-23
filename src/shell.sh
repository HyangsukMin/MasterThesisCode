#!/bin/bash
gpu_num=$1
dataset=$2

if [ "$dataset" = "mhealth" ]; then
    echo "DATASET is MHEALTH"
else
    echo "DATASET IS NOT MHEALTH"
fi

a=10
b=20

if [ $a == $b ]
then
   echo "a is equal to b"
elif [ $a -gt $b ]
then
   echo "a is greater than b"
elif [ $a -lt $b ]
then
   echo "a is less than b"
else
   echo "None of the condition met"
fi
