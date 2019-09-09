#!/bin/bash
for k in {10..100..10}
do
    for i in {0..4}
    do
        echo python train.py $1 $k -s $i -g $2
        python train.py $1 $k -s $i -g $2
    done
done
