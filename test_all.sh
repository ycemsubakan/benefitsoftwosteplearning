#!/bin/bash
for k in {10..100..10}
do
    for i in {0..4}
    do
        echo python test.py $1 $k $2 -s $i -g $3
        python test.py $1 $k $2 -s $i -g $3
    done
done
