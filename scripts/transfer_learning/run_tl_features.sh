#!/bin/bash
# Slurm PARAMETERS
#SBATCH --job-name=team10_tl
#SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --account=siads696s24_class
#SBATCH --mail-user mhuebsch@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-gpu=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/error-%x-%j.log

# run bashrc
source /home/mhuebsch/.bashrc
cd /home/mhuebsch/SIADS696_Milestone2/scripts/transfer_learning
# module load python3.9-anaconda/2021.11
module load cuda cudnn
conda activate ms2

# some fancy logging
START=`date +%s`; STARTDATE=`date`;
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] Starting the workflow
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] We got the following cores: $CUDA_VISIBLE_DEVICES
python create_tl_features.py -dev cuda -image_list /home/mhuebsch/SIADS696_Milestone2/data/BBBC021_v1_final_test -channels "DAPI,Tubulin,Actin"

EXITCODE=$?
# some fancy logging
END=`date +%s`; ENDDATE=`date`
echo [INFO] [$END] [$ENDDATE] [$$] [$JOB_ID] Workflow finished with code $EXITCODE
echo [INFO] [$END] [`date`] [$$] [$JOB_ID] Workflow execution time \(seconds\) : $(( $END-$START ))
