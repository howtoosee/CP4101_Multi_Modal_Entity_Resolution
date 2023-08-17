#!/bin/bash
#SBATCH --job-name=TEST-SOLVE
#SBATCH --partition=medium
#SBATCH --time=3:00:00
#SBATCH --output=out_test.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=v100:1

source activate fyp
#python3 src/main.py --data="/home/x/xchen/CP4101_BComp_Dissertation/data/laion/coco/dataframes/dataframe-img-soc-topic-frac=0.2.csv" --batch_size=256 --num_epochs=3
python3 src/main.py --data="coco2017" --is_multilabel --batch_size=64 --num_epochs=10 --max_samples=1000 --data_dir="/home/x/xchen/CP4101_BComp_Dissertation/data/coco-org/coco2017"
