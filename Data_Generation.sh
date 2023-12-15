#!/bin/bash
source ~/.bashrc
module load gcc/10.2.0
conda activate /home/gpalmari/.conda/envs/signature_JC_env
python /home/gpalmari/OnlineLearning/main_Data_Generation.py