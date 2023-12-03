#!/bin/zsh

#SBATCH --job-name=jupyter
#SBATCH --qos=cml-high
#SBATCH --account=cml-tokekar
#SBATCH --partition=cml-dpart
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --output=./logs/train-sac.log

source ~/.zshrc
conda activate procgen
python algorithms/sac_procgen.py --env-id starpilot --track --capture-video --total-timesteps 5000000
