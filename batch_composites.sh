#!/bin/bash
#SBATCH --gres=gpu:1 --partition=booster
#SBATCH --account=hhb19
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=composites
#SBATCH --output='/p/home/jusers/mueller29/juwels/jupyter_notebooks/composite_analysis/logs/%x_%j.out'
#SBATCH --error='/p/home/jusers/mueller29/juwels/jupyter_notebooks/composite_analysis/logs/%x_%j.err'

#SBATCH --mail-type=end      
#SBATCH --mail-type=fail
#SBATCH --mail-user=vasco.mueller@awi.de

#SBATCH --array=2015-2020 # job array index for each year


# Begin of section with executable commands

module --force purge
source ~/.bashrc
conda activate implicit_filter

cd /p/home/jusers/mueller29/juwels/jupyter_notebooks/composite_analysis/

var=$1

#with a total of 48 cores, 4 gpus, 512 GB memory this should allow to run 4 tasks in parallel

#srun python make_composite.py ${SLURM_ARRAY_TASK_ID} ${var}
#srun --cpus-per-task=12 --gres=gpu:1 --mem=128G python make_composite.py ${SLURM_ARRAY_TASK_ID} ${var}
python make_composite.py ${SLURM_ARRAY_TASK_ID} ${var}