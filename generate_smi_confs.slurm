#!/bin/bash
#SBATCH --job-name=TDConfs
#SBATCH --partition=dept_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate torsional_diffusion
which conda
echo $CONDA_DEFAULT_ENV
python generate_smi_confs.py --smi refined.smi --out all_conformers.smi --inference_steps 20 --tqdm --batch_size 128 --no_energy    
