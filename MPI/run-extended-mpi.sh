#!/bin/bash

## Specifies the interpreting shell for the job.
#$ -S /bin/bash

## Specifies that all environment variables active within the qsub utility be exported to the context of the job.
#$ -V

## Execute the job from the current working directory.
#$ -cwd

## Parallel programming environment (mpich) to instantiate and number of computing slots.
#$ -pe mpich 8
#$ -v  OMP_NUM_THREADS=8

## The  name  of  the  job.
#$ -N heat_hybrid

## Send an email at the start and the end of the job.
#$ -m be

## The email to send the queue manager notifications. 
#$ -M ova2@alumnes.udl.cat

## The folders to save the standard and error outputs.
#$ -o .
#$ -e .

MPICH_MACHINES=$TMPDIR/mpich_machines
cat $PE_HOSTFILE | awk '{print $1":"$2}' > $MPICH_MACHINES


## In this line you have to write the command that will execute your application.
mkdir MPI8-OMP8-$1-$2
mpiexec -f $MPICH_MACHINES -n $NSLOTS ./heat $1 $2 ./MPI8-OMP8-$1-$2/img-$1-$2.bmp


rm -rf $MPICH_MACHINES

