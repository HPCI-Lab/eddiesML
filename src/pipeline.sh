#!/bin/bash
#SBATCH --job-name=pipe_1_gpu
#SBATCH --output=pipe_1_gpu_hid_chan_64_epochs_50_lr_0.001.log

#SBATCH --partition=gpu #shared
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --account=ab0995          # Charge resources on this project account

#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=desta.gebremedhin@unitn.it

source ~/.bashrc
conda activate eddy-tracking-new

python3 ~/eddiesGNN/src/pipeline.py ~/eddiesGNN/src/config/pipeline.yaml
