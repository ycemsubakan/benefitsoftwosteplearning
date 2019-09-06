#!/bin/bash
for i in {0..4}
do
    echo python train.py $1 $2 -s $i -g $3
    python train.py $1 $2 -s $i -g $3
done
