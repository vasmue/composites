#!/bin/bash
#SBATCH --gres=gpu:1 --partition=booster
#SBATCH --account=hhb19
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=composites
#SBATCH --output='/p/home/jusers/mueller29/juwels/jupyter_notebooks/composite_analysis/logs/%x_%j.out'
#SBATCH --error='/p/home/jusers/mueller29/juwels/jupyter_notebooks/composite_analysis/logs/%x_%j.err'

#SBATCH --mail-type=end      
#SBATCH --mail-type=fail
#SBATCH --mail-user=vasco.mueller@awi.de

# Begin of section with executable commands

module --force purge
source ~/.bashrc
conda activate implicit_filter

cd /p/home/jusers/mueller29/juwels/jupyter_notebooks/composite_analysis/

python make_composite.py 2015 temp_100
