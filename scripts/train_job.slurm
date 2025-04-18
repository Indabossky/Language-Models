#!/bin/bash
#SBATCH --job-name=LangModel
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16GB

# Load Modules
module load python/3.10.4
module load cuda/11.7
module load cudnn/8.4.1
module load pytorch/1.12.1

# Run Python Script
python3 main.py
