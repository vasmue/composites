#!/bin/bash
#SBATCH --job-name="composites-pt1"
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --account=hhb19
#SBATCH --hint=nomultithread
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4

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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

var=$1

#with a total of 48 cores, 4 gpus, 512 GB memory this should allow to run 4 tasks in parallel

srun -n1 --exact --cpus-per-task=12 --gres=gpu:1 python make_composite.py 2015 ${var} &
srun -n1 --exact --cpus-per-task=12 --gres=gpu:1 python make_composite.py 2016 ${var} &
srun -n1 --exact --cpus-per-task=12 --gres=gpu:1 python make_composite.py 2017 ${var} &
srun -n1 --exact --cpus-per-task=12 --gres=gpu:1 python make_composite.py 2018 ${var} &
wait