#!/bin/bash

export OMP_NUM_THREADS=1

#----
date

list_Lx=(3 3 4 4)
list_Ly=(3 4 4 5)

for i in \
`seq 0 3`
do
  Lx=${list_Lx[${i}]}
  Ly=${list_Ly[${i}]}
  N=$((Lx*Ly))
  echo ${Lx} ${Ly} ${N}
  dir=dat_N${N}
  mkdir -p ${dir}
  python TFIsing_2d_nn_ED_with_MF_schedule.py -Lx ${Lx} -Ly ${Ly} > ${dir}/dat
#  python TFIsing_2d_nn_ED_with_MF_schedule.py -Lx ${Lx} -Ly ${Ly} > ${dir}/dat_N${N}
  mv fig*.png ${dir}
  date
done
