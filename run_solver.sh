#!/bin/bash
#SBATCH --job-name=SOLVE-FUKK
#SBATCH --partition=long
#SBATCH --time=71:59:59
#SBATCH --output=out_solve-br-full-2.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=v100:1

source activate fyp
python3 src/main.py --data="coco2017" --is_multilabel --batch_size=64 --num_epochs=20 --data_dir="/home/x/xchen/CP4101_BComp_Dissertation/data/coco-org/coco2017" --save_checkpoint="/home/x/xchen/CP4101_BComp_Dissertation/projects/main-project/checkpoints" --image_model="beit" --text_model="clip"
