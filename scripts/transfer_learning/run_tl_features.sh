#!/bin/bash
# Slurm script to run the transfer learning job.
# Uses one node + one gpu
# Slurm PARAMETERS
#SBATCH --job-name=team10_tl
#SBATCH --nodes=1
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
#SBATCH --time=06:30:00
#SBATCH --output=/home/%u/siads696/%x-%j.log
#SBATCH --error=/home/%u/siads696/error-%x-%j.log

# run image embedding creation as a cluster job on Great Lakes HPC
# using 1 node and 1 gpu

# run bashrc
source /home/mhuebsch/.bashrc
# switch to directory containing all scripts
cd /home/mhuebsch/SIADS696_Milestone2/scripts/transfer_learning
# load a python+conda module if necessary
# module load <ptyhon module>
module load cuda cudnn
# activate custom conda environment
conda activate ms2

START=`date +%s`; STARTDATE=`date`;
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] Starting the workflow
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] We got the following cores: $CUDA_VISIBLE_DEVICES

# set device to "cuda" to enable gpu usage
python create_tl_features.py -dev cuda \
	-image_list /home/mhuebsch/SIADS696_Milestone2/data/BBBC021_v1_final \
	-outdir /home/mhuebsch/siads696/data  \
	-channels "DAPI,Tubulin,Actin"

EXITCODE=$?

END=`date +%s`; ENDDATE=`date`
echo [INFO] [$END] [$ENDDATE] [$$] [$JOB_ID] Workflow finished with code $EXITCODE
echo [INFO] [$END] [`date`] [$$] [$JOB_ID] Workflow execution time \(seconds\) : $(( $END-$START ))
