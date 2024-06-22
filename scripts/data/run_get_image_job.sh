#!/bin/bash
# Slurm job to download all image archives
# unzip the archives
# remove the zip file after extraction
# Slurm PARAMETERS
#SBATCH --job-name=team10_gi
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1g
#SBATCH --account=siads696s24_class
#SBATCH --mail-user mhuebsch@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=standard
#SBATCH --time=00:40:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/error-%x-%j.log

# run the image retrieval and uppacking as a cluster job
# due to its run time. No blocking of resources on the login node

# run bashrc
source /home/mhuebsch/.bashrc
# actual image directory on the SIADS 696 scratch space
cd /scratch/siads696s24_class_root/siads696s24_class/mhuebsch/images

START=`date +%s`; STARTDATE=`date`;
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] Starting the workflow

# shell script in the image directory
./get_image_files.sh

EXITCODE=$?

END=`date +%s`; ENDDATE=`date`
echo [INFO] [$END] [$ENDDATE] [$$] [$JOB_ID] Workflow finished with code $EXITCODE
echo [INFO] [$END] [`date`] [$$] [$JOB_ID] Workflow execution time \(seconds\) : $(( $END-$START ))
