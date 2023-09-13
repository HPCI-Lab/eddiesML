#!/bin/bash
#SBATCH --job-name=pipeline       # Job name
#SBATCH --partition=shared        # Partition name
#SBATCH --output=pipeline.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=100G                 # Amount of memory needed
#SBATCH --time=00:05:00
#SBATCH --account=ab0995          # Charge resources on this project account

#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=massimiliano.fronza@unitn.it

source ~/.bashrc
conda activate eddy-tracking

python3 ~/eddiesGNN/src/pipeline.py ~/eddiesGNN/src/config/pipeline.yaml
