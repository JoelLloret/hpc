#!/bin/bash
#$ -S /bin/bash
#$ -V
#$ -cwd 
#$ -N heat_serial
#$ -m e
#$ -M ova2@alumnes.udl.cat

## In this line you have to write the command that will execute your application.

mkdir serial-$1-$2

./output $1 $2 ./serial-$1-$2/img-$1-$2.bmp

