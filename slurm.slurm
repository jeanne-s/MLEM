#!/bin/bash
#SBATCH --job-name=mamba         
#SBATCH --partition=cpu               # Take a node from the 'cpu' partition
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --cpus-per-task=40            # Ask for 10 CPU cores
##SBATCH --gres=gpu:1 
#SBATCH --mem=70G                    # Memory request; MB assumed if unit not specified
#SBATCH --time=20:00:00               # Time limit hrs:min:sec
#SBATCH --output=%x-%j.log            # Standard output and error log
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

echo "Running job on $(hostname)"

echo "computation start $(date)"
/scratch2/jsalle/.conda/envs/p11/bin/python3 pipeline_offset.py  
echo "computation end : $(date)"
