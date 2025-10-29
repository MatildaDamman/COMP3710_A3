#!/bin/bash
#SBATCH --job-name=final_attempt
#SBATCH --output=final_attempt_%j.log
#SBATCH --error=final_attempt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=p100
#SBATCH --gres=gpu:p100:1
#SBATCH --time=05:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s4705711@student.uq.edu.au

echo "🎯 FINAL ATTEMPT: 76.25% + Optimal Learning Rate"
echo "Job ID: $SLURM_JOB_ID"
echo "Last chance to reach 80%!"
echo "=========================================="

cd ~/comp3710/ISIC_Training/

echo "📦 Installing packages..."
python3 -m pip install --user torch torchvision pandas pillow numpy

echo "🔍 Environment check..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo "🚀 Starting final attempt..."
python3 final_attempt.py

echo "🏆 Final attempt completed at $(date)"