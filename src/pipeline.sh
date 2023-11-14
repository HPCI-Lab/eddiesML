#!/bin/bash
#SBATCH --job-name=pipe_7
#SBATCH --output=pipe_7_cpu_speed_test.log

#SBATCH --partition=gpu #shared
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --account=ab0995          # Charge resources on this project account

#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=massimiliano.fronza@unitn.it

source ~/.bashrc
conda activate eddy-tracking-new

python3 ~/eddiesGNN/src/pipeline.py ~/eddiesGNN/src/config/pipeline.yaml
