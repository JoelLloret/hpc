#!/bin/bash
#$ -S /bin/bash
#$ -V
#$ -cwd 
#$ -N heat_serial
#$ -m e
#$ -M ova2@alumnes.udl.cat
#$ -v  OMP_NUM_THREADS=8

## In this line you have to write the command that will execute your application.

mkdir OMP8-$1-$2


./output $1 $2 ./OMP8-$1-$2/img-$1-$2.bmp

