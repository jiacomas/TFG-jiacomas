#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -D .                     # Directori de treball (arrel del projecte)
#SBATCH -t 0-02:00               # Runtime: 2 hores de temps límit
#SBATCH -p tfg
#SBATCH --mem 8192               # Pugem a 8GB per seguretat amb el dataset
#SBATCH -o logs/%x_%j.out        # Output organitzat en carpeta logs
#SBATCH -e logs/%x_%j.err        # Errors organitzats
#SBATCH --gres gpu:1

# --- Environment Setup ---

# 1. Initialize Conda for this shell session
# This path points to your specific miniconda installation
source /fhome/jccomas/miniconda3/etc/profile.d/conda.sh

# 2. Activate your environment by path
conda activate /fhome/jccomas/miniconda3/envs/tfg

# 3. Weights & Biases configuration
export WANDB_API_KEY="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"
export WANDB_MODE="online"

# 4. Add the current directory to PYTHONPATH
# This ensures "import src" works correctly
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Job started on $(hostname) at $(date)"
echo "Using Python: $(which python)"

# --- Execution ---

# Check GPU status before starting
nvidia-smi

# Run the training script
python3 src/models/ml_classic.py
# python3 src/models/deep_learning.py

echo "Job finished at $(date)"
