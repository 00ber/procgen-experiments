#!/bin/zsh

#SBATCH --job-name=jupyter
#SBATCH --qos=cml-high
#SBATCH --account=cml-tokekar
#SBATCH --partition=cml-dpart
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --output=./logs/train-dqn.log

source ~/.zshrc
conda activate procgen
python algorithms/dqn_procgen.py --env-id starpilot --num-envs 1 --track --capture-video --total-timesteps 5000000
