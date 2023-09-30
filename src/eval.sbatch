#!/bin/bash
#SBATCH --job-name="gwm_davis_num_qry_4"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=64G
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=DEADLINE
#SBATCH --comment="DEADLINE."
#SBATCH --output=/home/stud/mahmo/slurm/logs/slurm-%j.out
#SBATCH --error=/home/stud/mahmo/slurm/logs/slurm-%j.out
conda run -n gwm python /home/stud/mahmo/temp/guess-what-moves/src/main_ssl.py --resume_path ../outputs/stv2/20230705_004149_te/checkpoints/checkpoint_0019999.pth GWM.DATASET STv2
# Comment (Magdy Mahmoud)
# Test of kmeans on coco