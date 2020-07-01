#!/bin/bash

export OMP_NUM_THREADS=1

#----
date

for N in \
4 5 6 7 8 9 10 11 12 13 14 15 16
do
  dir=dat_N${N}
  mkdir -p ${dir}
  python TFIsing_1d_nn_ED_with_MF_schedule.py -N ${N} > ${dir}/dat
  mv fig*.png ${dir}
  date
done
