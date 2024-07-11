#!/bin/bash -l

#SBATCH --job-name="train_lstm"
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.err

# walltime (hh:mm:ss)
#SBATCH -t 15:00:00

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

## email adress
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicolas.caronprowork@gmail.com

module purge
module load miniconda3@4.10.3/gcc-12.1.0
conda activate train_lstm_env
#python train/train_ConvLSTM.py
python train/train_LSTM_pytorch.py