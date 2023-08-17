#!/bin/bash
#SBATCH --job-name=FEWSHOT
#SBATCH --partition=long
#SBATCH --time=71:59:59
#SBATCH --output=out_fs.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=v100:1

source activate fyp
python3 src/fewshot.py
