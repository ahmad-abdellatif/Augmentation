#!/bin/bash
#SBATCH --account=def-daslab
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=10G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-1:00
#SBATCH --mail-user=ahmad.abdellatif87@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

ulimit -n 2048
module purge
module load cuda cudnn
module load python/3 scipy-stack
module load gcc/9.3.0 arrow/2.0.0
source ~/Simpletransformers_Env/bin/activate
python /home/ahmad2/repo/TrainingDataAugmentation/scripts/paraphrasing/BART/original_BART.py