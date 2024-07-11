#!/bin/bash -l

#SBATCH --job-name="train_lightGBM"
#SBATCH --output=log/%x.%J.out
#SBATCH --error=log/%x.%J.err

# walltime (hh:mm:ss)
#SBATCH -t 24:00:00

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

## email adress
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicolas.caronprowork@gmail.com

module purge
module load miniconda3@4.10.3/gcc-12.1.0
conda activate train_env
python train/train_lightGBM.py