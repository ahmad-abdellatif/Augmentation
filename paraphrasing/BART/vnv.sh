#!/bin/bash
#SBATCH --account=def-daslab
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=64G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=00-20:40
#SBATCH --mail-user=ahmad.abdellatif87@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL


module load python/3.6.10
ulimit -n 2048
module load StdEnv/2020
module load gcc/8.4.0
module load cuda/10.2
source ~/work_venv/bin/activate
python /home/ahmad2/repo/TrainingDataAugmentation/scripts/paraphrasing/BART/original_BART.py