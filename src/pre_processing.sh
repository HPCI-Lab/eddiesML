#!/bin/bash
#SBATCH --job-name=pre_processing	# Job name
#SBATCH --partition=shared         	# Partition name
#SBATCH --mem=32G			# Amount of memory needed
#SBATCH --time=00:30:00
#SBATCH --account=ab0995		# Charge resources on this project account
#SBATCH --output=pre_processing.log	# Log name

# Un-comment these 3 lines and put your email to be notified of when the processing ends
##SBATCH --mail-type=end
##SBATCH --mail-type=fail
##SBATCH --mail-user=massimiliano.fronza@unitn.it

source ~/.bashrc
conda activate eddy-tracking

python3 ~/eddiesGNN/src/pre_processing.py ~/eddiesGNN/src/config.yaml
