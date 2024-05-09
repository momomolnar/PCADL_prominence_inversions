#!/bin/bash -l

#PBS -N test_set
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=16:ngpus=1:mem=256gb
#PBS -l walltime=06:00:00

module load conda/latest
module load cuda/11.7.1
conda activate pytorch_env

cd /glade/u/home/mmolnar/Projects/PROPCA/
python3 -m pca -checkpoint  /glade/u/home/mmolnar/Projects/PROls *PCA/laughing-happiness