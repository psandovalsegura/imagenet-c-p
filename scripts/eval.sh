#!/bin/bash
#SBATCH --account=djacobs
#SBATCH --job-name=im-c
#SBATCH --time=1-00:00:00
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu
#--SBATCH --dependency=afterok:

set -x

export IMAGENET_DIR='/vulcanscratch/psando/val'
export IMAGENET_C_DIR='/fs/vulcan-projects/stereo-detection/imagenet-c/'
export MODEL_DIR='/vulcanscratch/psando/TorchHub'

python ImageNet-C/test.py --ngpu 1\
                          --model-name 'resnet18'\

