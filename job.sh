#!/bin/bash
 
#PBS -P na4
#PBS -q gpuvolta
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l mem=32GB
#PBS -l jobfs=16GB
#PBS -l walltime=01:00:00
#PBS -l wd
#PBS -o /home/563/st8309/qaml/job.out
#PBS -e /home/563/st8309/qaml/job.err
 
# Load module, always specify version number.
module load python3/3.9.2
module load cuda/11.4.1
 
# Set number of OMP threads
export OMP_NUM_THREADS=$PBS_NCPUS
 
# Must include `#PBS -l storage=scratch/ab12+gdata/yz98` if the job
# needs access to `/scratch/ab12/` and `/g/data/yz98/`. Details on
# https://opus.nci.org.au/display/Help/PBS+Directives+Explained.
 
# Run Python applications
python3 test.py > $PBS_JOBID.log