#!/bin/bash
#SBATCH --job-name=pipe
#SBATCH --output=pipeline.log

#SBATCH --partition=shared        # Partition name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G                 # Amount of memory needed
#SBATCH --time=08:00:00
#SBATCH --account=ab0995          # Charge resources on this project account

#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=massimiliano.fronza@unitn.it

source ~/.bashrc
conda activate eddy-tracking

python3 ~/eddiesGNN/src/pipeline.py ~/eddiesGNN/src/config/pipeline.yaml
