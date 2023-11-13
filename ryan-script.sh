#!/bin/bash

values=(0.1 0.25 0.5 0.75 1 1.25 1.5)
pvals=(10 25 50 75)
rvals=(0.2 0.4 0.6 0.8 1 1.2)
# values=(0.1)
# pvals=(10)
# rvals=(0.2)

echo "~~~~~ Mathias Sampling: ~~~~~" > "gosdt/sample_out.txt"
for value in "${values[@]}"
do
    for p in "${pvals[@]}"
    do
        python gosdt/ryan-test.py -q "$value" -p "$p" --sampling_method mathias >> gosdt/sample_out.txt
        echo -e "\n" >> "gosdt/sample_out.txt"
    done
done
echo -e "\n\n" >> "gosdt/sample_out.txt"
echo "~~~~~ GOSDTwGSampling: ~~~~~" >> "gosdt/sample_out.txt"
for value in "${values[@]}"
do
    for r in "${rvals[@]}"
    do
        python gosdt/ryan-test.py -q "$value" -r "$r" --sampling_method gosdtwG >> gosdt/sample_out.txt
    done
done