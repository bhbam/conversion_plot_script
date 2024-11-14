#!/bin/bash

for i in {0..9}; do
    nohup python Run_3_convert_biased_parquey_to_unbiased_parquet.py -m m1p2To3p6 -n $i > unbiased_m1p2To3p6_$i.log  2>&1 &
done
